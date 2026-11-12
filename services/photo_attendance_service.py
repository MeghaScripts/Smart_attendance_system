"""
Photo Attendance Service — mark attendance from a single class photo.
Teacher uploads one group photo → system finds all faces → marks them present.

Tuned for top-mounted CCTV cameras where faces appear tilted downward, and
for group photos of up to EXPECTED_CLASS_SIZE (default 60) students.
"""
import cv2
import numpy as np

from core.config import face_settings
from services.embedding_service import FaceEmbeddingService


class PhotoAttendanceService:
    def __init__(self):
        self.embedding_service = FaceEmbeddingService()

    def _preprocess_photo(self, img: np.ndarray, expected_faces: int = None) -> np.ndarray:
        """
        Upscale small images so RetinaFace can detect faces reliably.

        A flat 1280px floor works for a handful of people but leaves each
        face at ~15-25px in a 60-person classroom photo — well under
        MIN_FACE_SIZE. Scale the target width with the expected headcount
        instead, so a 60-student photo gets meaningfully more resolution
        per face than a 5-student one.
        """
        expected_faces = expected_faces or face_settings.EXPECTED_CLASS_SIZE
        h, w = img.shape[:2]

        # Rough heuristic: assume faces are laid out roughly in a grid, so
        # resolution-per-face scales with sqrt(expected_faces). Floor of
        # 1280 for small photos, scaling up toward ~3200px for 60 students.
        target_w = max(1280, int(320 * (expected_faces ** 0.5)))

        if w < target_w:
            scale = target_w / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            print(f"   📐 Upscaled to {img.shape[1]}x{img.shape[0]} (target_w={target_w} for ~{expected_faces} faces)")
        return img

    def process_class_photo(self, image_bytes: bytes, class_id: str, session_id: str) -> dict:
        """
        Runs the SAME recognize_face_in_frame() as webcam and CCTV, so all three
        sources agree and the group photo feeds the teacher review dashboard
        exactly like they do:
          * confidently recognised faces        -> returned in "matched"
          * everything else that was detected   -> sent to the review dashboard
            (blurry, too small, unmatched, ambiguous between students)
          * students already marked this session are skipped.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        embeddings = self.embedding_service.get_class_embeddings(class_id)
        if not embeddings:
            raise ValueError("No students enrolled for face recognition in this class")

        expected_faces = max(len(embeddings), face_settings.EXPECTED_CLASS_SIZE)
        img = self._preprocess_photo(img, expected_faces=expected_faces)

        print(f"📸 [PHOTO-UPLOAD] Processing class photo for {class_id}...")
        print(f"   Image: {img.shape[1]}x{img.shape[0]}, model={self.embedding_service.model_name}, "
              f"enrolled={len(embeddings)}")

        svc = self.embedding_service
        matches, unmatched_count = svc.recognize_face_in_frame(
            img, embeddings, class_id=class_id,
            check_liveness=False, is_cctv=False,
            session_id=session_id, source="photo",
            threshold_delta=face_settings.PHOTO_THRESHOLD_DELTA,
        )

        matched, seen = [], set()
        for student_id, student_name, email, confidence, _live in matches:
            if student_id in seen:
                continue
            seen.add(student_id)
            matched.append({
                "student_id": student_id,
                "student_name": student_name,
                "email": email,
                "confidence": confidence,
            })

        stats = getattr(svc, "last_stats", {}) or {}
        return {
            "total_faces_detected": stats.get("faces", len(matches) + unmatched_count),
            "matched": matched,
            "unmatched_faces": unmatched_count,
            "sent_to_review": stats.get("sent_to_review", 0),
            "already_marked": stats.get("already_marked", 0),
        }
