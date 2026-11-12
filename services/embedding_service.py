import base64
import os
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

import db
from core.config import face_settings
from services.liveness_service import liveness_detector
from services.quality_service import (
    quality_info,
    yaw_ratio,
    make_degraded_variants,
    DEGRADED_VARIANTS_PER_PHOTO,
    MIN_FACE_PX_HARD,
    YAW_HARD,
)

# Low-quality (blurry / small / angled) faces are still recognised, but must
# clear a STRICTER distance than good faces. Tune via .env.
LOW_QUALITY_THRESHOLD_TIGHTEN = float(os.getenv("LOW_QUALITY_THRESHOLD_TIGHTEN", 0.03))

# Teacher review feed ----------------------------------------------------
# Rejected faces smaller than this (px) are useless to a human, so skip them.
REVIEW_MIN_REJECT_PX = int(os.getenv("REVIEW_MIN_REJECT_PX", 16))

_review_sink = None

# When a class has a trained SVM, a face is only AUTO-marked if the SVM and the
# cosine distance agree on the same student. Any disagreement (SVM says A but
# cosine fails, or cosine says A but the SVM says B) goes to the teacher.
REQUIRE_MODEL_AGREEMENT = os.getenv("REQUIRE_MODEL_AGREEMENT", "1") == "1"


def _parse_grid(spec) -> Tuple[int, int]:
    try:
        c, r = str(spec).lower().split("x")
        return max(1, int(c)), max(1, int(r))
    except Exception:
        return 1, 1


# CCTV (high corner camera): faces are small, so besides the full frame we also
# scan overlapping, up-scaled tiles. "3x2" = 6 tiles (3 across, 2 down); "1x1" turns tiling off.
CCTV_TILES = _parse_grid(os.getenv("CCTV_TILES", "3x2"))
CCTV_TILE_UPSCALE = float(os.getenv("CCTV_TILE_UPSCALE", 1.5))
CCTV_TILE_OVERLAP = float(os.getenv("CCTV_TILE_OVERLAP", 0.2))


def _box(fa: dict):
    return fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)


def _overlaps(a: dict, b: dict) -> bool:
    """True if two face boxes are (mostly) the same face."""
    ax, ay, aw, ah = _box(a)
    bx, by, bw, bh = _box(b)
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return False
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    smaller = max(1, min(aw * ah, bw * bh))
    return inter / max(1, union) > 0.3 or inter / smaller > 0.6

# How many closest students to show on a review card
REVIEW_TOP_CANDIDATES = int(os.getenv("REVIEW_TOP_CANDIDATES", 3))
# a face this close (cosine) to a student counts as "looks like them"; same cap the review queue uses
REVIEW_CLOSE_DIST = float(os.getenv("REVIEW_MAX_DIST", 0.55))


def set_review_sink(fn):
    """main.py registers a callable that receives ONE dict:
         {session_id, class_id, crop, candidates, info, reason, embedding, source}
       candidates = [{"student_id", "student_name", "distance"}, ...] best first.
    Every recognition path (webcam, CCTV, photo) goes through
    recognize_face_in_frame, so one sink covers them all."""
    global _review_sink
    _review_sink = fn


def _emit_review(class_id, session_id, crop, candidates, distance_unused, info, reason,
                 embedding=None, source=None, marked_near=None):
    if _review_sink is None or crop is None or class_id is None:
        return
    try:
        _review_sink({
            "session_id": session_id, "class_id": class_id, "crop": crop,
            "candidates": candidates or [], "info": info, "reason": reason,
            "embedding": embedding, "source": source, "marked_near": marked_near or [],
        })
    except Exception as e:
        print(f"⚠️  review sink error: {e}")


def _emit_already_marked(class_id, session_id, student, source):
    """Tell the review dashboard that a face of an already-marked student was
    seen again (shown as an alert, not as a card)."""
    if _review_sink is None or class_id is None or not session_id or not student:
        return
    try:
        from services.session_manager import session_manager
        if session_manager.is_absent_locked(session_id, student["student_id"]):
            return          # teacher set this student absent on purpose: no alert, no re-marking
    except Exception:
        pass
    try:
        _review_sink({
            "kind": "already_marked", "session_id": session_id, "class_id": class_id,
            "student_id": student["student_id"], "student_name": student.get("student_name"),
            "source": source,
        })
    except Exception as e:
        print(f"⚠️  review sink error: {e}")


def _reject_reason(info: dict) -> str:
    if info.get("width", 0) < MIN_FACE_PX_HARD:
        return "too_small"
    if info.get("yaw", 0) > YAW_HARD:
        return "extreme_angle"
    return "too_blurry"


def _with_cell(info: dict, frame_shape: tuple, facial_area: dict) -> dict:
    """Add a coarse 6x6 grid cell of the face position (≈ the student's seat),
    used to dedupe review cards for faces with no suggested name."""
    fh, fw = frame_shape[:2]
    cx = facial_area.get("x", 0) + facial_area.get("w", 0) / 2.0
    cy = facial_area.get("y", 0) + facial_area.get("h", 0) / 2.0
    info = dict(info)
    info["cell"] = f"{min(5, int(cx / max(fw, 1) * 6))}_{min(5, int(cy / max(fh, 1) * 6))}"
    return info


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Return original + up to 11 augmented variants (12 total, configurable via
    AUGMENTATION_VARIANTS) for robust multi-embedding enrollment from a
    single captured/uploaded photo. Includes downward-tilt variants for
    top-mounted CCTV cameras.
    """
    augmented = [img]
    h, w = img.shape[:2]

    # 1. CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    augmented.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    # 2. Brighter
    augmented.append(cv2.convertScaleAbs(img, alpha=1.0, beta=30))

    # 3. Darker
    augmented.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-30))

    # 4. Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # 5. Left rotation +5
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # 6. Right rotation -5
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -5, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # 7. Slight blur
    augmented.append(cv2.GaussianBlur(img, (3, 3), 0))

    # 8. Mild downward tilt ~15deg (top-mounted CCTV)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift = int(w * 0.08)
    dst_mild = np.float32([[shift, 0], [w - shift, 0], [w, h], [0, h]])
    M_mild = cv2.getPerspectiveTransform(src_pts, dst_mild)
    augmented.append(cv2.warpPerspective(img, M_mild, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # 9. Moderate downward tilt ~25deg
    shift2 = int(w * 0.15)
    dst_mod = np.float32([[shift2, 0], [w - shift2, 0], [w, h], [0, h]])
    M_mod = cv2.getPerspectiveTransform(src_pts, dst_mod)
    augmented.append(cv2.warpPerspective(img, M_mod, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # 10. Mild tilt + flip
    augmented.append(cv2.flip(
        cv2.warpPerspective(img, M_mild, (w, h), borderMode=cv2.BORDER_REPLICATE), 1
    ))

    # 11. Scale down 85% (student farther from camera)
    scale = 0.85
    rsz = cv2.resize(img, (int(w * scale), int(h * scale)))
    px = (w - rsz.shape[1]) // 2
    py = (h - rsz.shape[0]) // 2
    padded = cv2.copyMakeBorder(rsz, py, h - rsz.shape[0] - py,
                                 px, w - rsz.shape[1] - px, cv2.BORDER_REPLICATE)
    augmented.append(padded)

    return augmented[: max(1, face_settings.AUGMENTATION_VARIANTS_PER_PHOTO)]


# ---------------------------------------------------------------------------
# Quality-gate helpers (shared by enrollment + every recognition source)
# ---------------------------------------------------------------------------

def _facial_area_is_valid(facial_area: dict, frame_shape: tuple) -> Tuple[bool, str]:
    """
    Rejects the two failure modes that cause 'whole frame recognized as a
    face' and 'tiny noise detected as a face':
      - box covers most of the frame  → detector silently fell back to the
        full image instead of raising
      - box smaller than MIN_FACE_SIZE → too small to embed reliably
    """
    fh, fw = frame_shape[:2]
    w = facial_area.get("w", 0)
    h = facial_area.get("h", 0)

    if w <= 0 or h <= 0:
        return False, "zero_size"

    if w < face_settings.MIN_FACE_SIZE or h < face_settings.MIN_FACE_SIZE:
        return False, f"too_small ({w}x{h}px)"

    frame_area = max(fw * fh, 1)
    face_area_ratio = (w * h) / frame_area
    if face_area_ratio > face_settings.MAX_FACE_AREA_RATIO:
        return False, f"whole_frame_fallback ({face_area_ratio:.0%} of frame)"

    return True, "ok"


def _is_too_blurry(gray_roi: np.ndarray) -> bool:
    """Legacy hard blur check. NOT used by detect_and_embed_faces any more
    (replaced by the tiered quality_info gate); kept so other modules that
    import it don't break."""
    if gray_roi is None or gray_roi.size == 0:
        return True
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var() < face_settings.MIN_BLUR_VARIANCE


def _crop_from_facial_area(frame: np.ndarray, facial_area: dict) -> Optional[np.ndarray]:
    fh, fw = frame.shape[:2]
    x = max(0, facial_area.get("x", 0))
    y = max(0, facial_area.get("y", 0))
    x2 = min(fw, x + facial_area.get("w", 0))
    y2 = min(fh, y + facial_area.get("h", 0))
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]


def _crop_with_margin(frame: np.ndarray, facial_area: dict, margin: float = 0.25) -> Optional[np.ndarray]:
    """Face crop with some context around it. Used for the teacher review
    cards so a human can recognise the person."""
    fh, fw = frame.shape[:2]
    x = facial_area.get("x", 0)
    y = facial_area.get("y", 0)
    w = facial_area.get("w", 0)
    h = facial_area.get("h", 0)
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(fw, x + w + mx), min(fh, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _extract_embedding(img: np.ndarray, model_name: str, detector: str) -> Optional[np.ndarray]:
    """
    Enrollment-time embedding extraction. Enrollment photos are curated,
    single-face, teacher-supplied images, so a controlled fallback to
    enforce_detection=False is acceptable here — but the returned box is
    still validated against a whole-frame fallback before being trusted.
    """
    from deepface import DeepFace
    chain = [detector]
    if detector not in ("mtcnn", "opencv"):
        chain += ["mtcnn", "opencv"]
    elif detector != "opencv":
        chain += ["opencv"]

    for det in chain:
        try:
            result = DeepFace.represent(
                img_path=img, model_name=model_name,
                detector_backend=det, enforce_detection=True,
            )
            if result:
                facial_area = result[0].get("facial_area", {})
                ok, _ = _facial_area_is_valid(facial_area, img.shape)
                if ok:
                    return np.array(result[0]["embedding"])
        except Exception:
            pass

    for det in chain:
        try:
            result = DeepFace.represent(
                img_path=img, model_name=model_name,
                detector_backend=det, enforce_detection=False,
            )
            if result:
                facial_area = result[0].get("facial_area", {})
                ok, _ = _facial_area_is_valid(facial_area, img.shape)
                if ok:
                    return np.array(result[0]["embedding"])
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FaceEmbeddingService:
    """
    Pipeline implemented here:

        student photo → 12 augmented variants (+ CCTV-style degraded
        variants) → FaceNet512 → 512-D embeddings (stored in MySQL) → SVM
        classifier trained on top (classifier_service.py) → recognition =
        SVM prediction, cross-checked against cosine similarity + margin
        before being trusted.

    This class owns embedding extraction, storage, and the cosine-distance
    matcher (used as the verification step and as the fallback when a
    class doesn't have a trained classifier yet). SVM training/
    prediction lives in services/classifier_service.py.
    """

    def __init__(self):
        self.model_name = face_settings.FACE_MODEL
        self.detector = face_settings.FACE_DETECTOR
        self.threshold = face_settings.FACE_DISTANCE_THRESHOLD
        self.margin = face_settings.MATCH_MARGIN
        self.top_k = face_settings.MATCH_TOP_K_EMBEDDINGS

    # ── Enroll ──────────────────────────────────────────────────────────────

    def _check_similar_to_others(self, student_id: str, new_embeddings: List[List[float]]) -> Optional[dict]:
        """
        Compares the newly captured embeddings against every OTHER student
        already enrolled in the SAME class (cross-class similarity doesn't
        matter — recognition only ever compares within one class).

        Uses the SAME top-K mean-distance scoring as real recognition
        (match_embedding_to_students), not a raw minimum over every
        pairwise comparison. With up to MAX_ENROLLMENT_IMAGES x
        AUGMENTATION_VARIANTS_PER_PHOTO embeddings per student on both
        sides (e.g. 48), a raw minimum over thousands of pairwise
        comparisons trends lower purely from volume — by chance, at least
        one of thousands of noisy comparisons dips below threshold even
        between two genuinely different people. Averaging each new
        embedding's top-K closest distances to a candidate student (then
        averaging across all new embeddings) only flags a real look-alike
        risk — the same signal real recognition would actually produce.

        Returns None if no one in the class scores that close, else
        {"student_id", "distance"} for the closest look-alike.
        """
        student = db.fetch_one("SELECT class_id FROM students WHERE student_id = %s", (student_id,))
        if not student:
            return None
        class_id = student["class_id"]

        others = db.fetch_all(
            """
            SELECT fe.student_id, fe.embeddings
            FROM face_embeddings fe
            JOIN students s ON s.student_id = fe.student_id
            WHERE s.class_id = %s AND fe.student_id != %s
            """,
            (class_id, student_id),
        )
        if not others:
            return None

        new_arrs = [np.array(e) for e in new_embeddings]

        closest_id, closest_score = None, None
        for row in others:
            try:
                stored = [np.array(e) for e in db.from_json(row["embeddings"])]
            except Exception:
                continue
            if not stored:
                continue

            per_embedding_scores = []
            for ne in new_arrs:
                dists = sorted(self._cosine_distance(ne, se) for se in stored)
                top_k = dists[: self.top_k]
                per_embedding_scores.append(sum(top_k) / len(top_k))

            student_score = sum(per_embedding_scores) / len(per_embedding_scores)
            if closest_score is None or student_score < closest_score:
                closest_score = student_score
                closest_id = row["student_id"]

        if closest_id is not None and closest_score < self.threshold:
            return {"student_id": closest_id, "distance": round(float(closest_score), 4)}
        return None

    def enroll_student(self, student_id: str, image_bytes_list: List[bytes]) -> dict:
        """
        Enrolls a student from MIN_ENROLLMENT_IMAGES-MAX_ENROLLMENT_IMAGES
        real photos (default 4-10). Real photos captured in genuinely
        different conditions (glasses on/off, different lighting, slight
        angle, with or without a beard) give the classifier real appearance
        variation to average over — something synthetic augmentation alone
        can't provide. Each accepted real photo is ALSO expanded into
        AUGMENTATION_VARIANTS_PER_PHOTO synthetic variants for extra
        robustness to camera/lighting conditions, plus
        DEGRADED_VARIANTS_PER_PHOTO CCTV-style degraded variants (distance
        blur, motion blur, JPEG compression) so blurry CCTV faces land close
        to the stored embeddings.

        Total embeddings stored = num_valid_photos x (clean + degraded).
        The min/max photo limits apply to real photos, not to the synthetic
        variants generated from them. Clean embeddings are stored FIRST.
        """
        min_n = face_settings.MIN_ENROLLMENT_IMAGES
        max_n = face_settings.MAX_ENROLLMENT_IMAGES

        if len(image_bytes_list) > max_n:
            raise ValueError(
                f"Too many photos: got {len(image_bytes_list)}, maximum is {max_n}."
            )

        print(f"🔍 Enrolling {student_id} from {len(image_bytes_list)} real photo(s)...")

        clean_embeddings: List[List[float]] = []
        degraded_embeddings: List[List[float]] = []
        valid_photo_count = 0
        for idx, image_bytes in enumerate(image_bytes_list):
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  photo {idx + 1}/{len(image_bytes_list)} ❌ could not decode")
                continue

            variants = _augment_image(img)
            photo_embeddings = []
            for v_idx, variant in enumerate(variants):
                emb = _extract_embedding(variant, self.model_name, self.detector)
                if emb is not None:
                    photo_embeddings.append(emb.tolist())

            if not photo_embeddings:
                print(f"  photo {idx + 1}/{len(image_bytes_list)} ❌ no face detected in any variant — skipped")
                continue

            clean_embeddings.extend(photo_embeddings)
            valid_photo_count += 1

            # CCTV-style degraded copies (only for photos that already passed)
            photo_degraded = 0
            if DEGRADED_VARIANTS_PER_PHOTO > 0:
                for dv in make_degraded_variants(img, DEGRADED_VARIANTS_PER_PHOTO):
                    emb = _extract_embedding(dv, self.model_name, self.detector)
                    if emb is not None:
                        degraded_embeddings.append(emb.tolist())
                        photo_degraded += 1

            print(
                f"  photo {idx + 1}/{len(image_bytes_list)} ✅ "
                f"{len(photo_embeddings)}/{len(variants)} variant(s) + "
                f"{photo_degraded}/{DEGRADED_VARIANTS_PER_PHOTO} degraded embedded"
            )

        if valid_photo_count < min_n:
            raise ValueError(
                f"Only {valid_photo_count} of {len(image_bytes_list)} photo(s) had a usable face "
                f"— need at least {min_n} valid photos. Use clear, front-facing, well-lit photos, "
                f"and try uploading a few more (e.g. with/without glasses, slightly different angle)."
            )

        embeddings = clean_embeddings + degraded_embeddings
        print(
            f"  ✅ {valid_photo_count} valid photo(s) → {len(embeddings)} total embeddings for {student_id} "
            f"({len(clean_embeddings)} clean + {len(degraded_embeddings)} degraded)"
        )

        # Look-alike check uses CLEAN embeddings only, so noisy degraded
        # copies can't trigger false "too similar" blocks.
        similar = self._check_similar_to_others(student_id, clean_embeddings)
        if similar:
            print(
                f"  ⚠️  ALERT: {student_id}'s face is too similar to already-enrolled "
                f"{similar['student_id']} (distance={similar['distance']}, "
                f"threshold={self.threshold}) — BLOCKING enrollment to avoid recognition confusion."
            )
            return {
                "student_id": student_id,
                "enrolled": False,
                "blocked": True,
                "similar_to": similar,
                "warning": (
                    f"Enrollment blocked: this face is too similar to already-enrolled "
                    f"student {similar['student_id']} (distance={similar['distance']}, "
                    f"threshold={self.threshold}). Recognition would likely confuse these "
                    f"two people. Use clearer/different photos, or verify this isn't a "
                    f"duplicate enrollment of the same person under a different ID."
                ),
            }

        db.execute(
            """
            INSERT INTO face_embeddings
                (student_id, embeddings, embedding, model, num_embeddings, enrolled_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                embeddings        = VALUES(embeddings),
                embedding         = VALUES(embedding),
                model             = VALUES(model),
                num_embeddings    = VALUES(num_embeddings),
                updated_at        = NOW()
            """,
            (
                student_id,
                db.to_json(embeddings),
                db.to_json(embeddings[0]),
                self.model_name,
                len(embeddings),
                datetime.utcnow(),
            ),
        )

        return {
            "student_id": student_id,
            "enrolled": True,
            "embeddings_stored": len(embeddings),
            "degraded_embeddings": len(degraded_embeddings),
        }

    # ── Load enrolled embeddings for a class ────────────────────────────

    def get_class_embeddings(self, class_id: str) -> List[dict]:
        """
        Returns [{student_id, student_name, email, embeddings:[np.array,...]}]
        for every enrolled student in the class. Used both by the cosine
        matcher directly and by classifier_service.py to build SVM training data.
        """
        students = db.fetch_all(
            "SELECT student_id, student_name, email FROM students WHERE class_id = %s",
            (class_id,),
        )
        print(f"📋 {len(students)} students in {class_id}")
        if not students:
            return []

        student_ids = [s["student_id"] for s in students]
        placeholders = ",".join(["%s"] * len(student_ids))
        rows = db.fetch_all(
            f"SELECT student_id, embeddings, embedding FROM face_embeddings "
            f"WHERE student_id IN ({placeholders})",
            tuple(student_ids),
        )
        emb_by_student = {r["student_id"]: r for r in rows}

        expected_dim = 512 if "512" in self.model_name else 128
        embeddings = []
        for s in students:
            row = emb_by_student.get(s["student_id"])
            if not row:
                continue
            raw_embs = db.from_json(row.get("embeddings")) or [db.from_json(row.get("embedding"))]
            emb_arrays = [np.array(e) for e in raw_embs if e is not None]
            emb_arrays = [e for e in emb_arrays if len(e) == expected_dim]

            if not emb_arrays:
                print(f"  ⚠️  {s['student_name']} has no valid {expected_dim}-dim embeddings — re-enroll")
                continue

            embeddings.append({
                "student_id": s["student_id"],
                "student_name": s.get("student_name", "Unknown"),
                "email": s.get("email", ""),
                "embeddings": emb_arrays,
            })

        print(f"🧠 {len(embeddings)} students with valid embeddings loaded")
        return embeddings

    # ── CCTV / low-quality frame preprocessing ──────────────────────────────

    def preprocess_cctv_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess top-down/angled CCTV frame before face detection.
        - Upscales so small faces are detectable
        - CLAHE contrast for class window glare
        - Mild sharpening for RTSP compression blur
        """
        h, w = frame.shape[:2]
        scale = max(1.0, 1920 / max(h, w))
        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"   📐 Upscaled {w}x{h} → {new_w}x{new_h}")

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        kernel = np.array([[0, -0.5, 0],
                            [-0.5, 3, -0.5],
                            [0, -0.5, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        return frame

    # ── Rejected-face collection (for the teacher review page) ────────────

    @staticmethod
    def _collect_rejected(rejected_out, frame, face_data, facial_area, reason):
        if min(facial_area.get("w", 0), facial_area.get("h", 0)) < REVIEW_MIN_REJECT_PX:
            return  # too tiny for a human to recognise either
        crop = _crop_with_margin(frame, facial_area)
        roi = _crop_from_facial_area(frame, facial_area)
        if crop is None or roi is None or roi.size == 0:
            return
        info = _with_cell(quality_info(roi, det_conf=1.0, yaw=0.0), frame.shape, facial_area)
        rejected_out.append({
            "embedding": np.array(face_data["embedding"]),
            "facial_area": facial_area,
            "reason": reason,
            "crop": crop,
            "info": info,
        })

    # ── Shared detect + embed (used identically by webcam, CCTV, photo) ─────

    def _detect_tiled(self, frame: np.ndarray, detector: str, grid: Tuple[int, int]) -> List[dict]:
        """Scan overlapping, up-scaled tiles so small / far faces are found.
        Returns DeepFace-style dicts with facial_area in FULL-FRAME coordinates."""
        from deepface import DeepFace
        H, W = frame.shape[:2]
        cols, rows = grid
        ov = min(0.5, max(0.0, CCTV_TILE_OVERLAP))
        tw = W / (cols - (cols - 1) * ov)
        th = H / (rows - (rows - 1) * ov)
        sx = tw * (1 - ov)
        sy = th * (1 - ov)
        sc = CCTV_TILE_UPSCALE
        out = []
        for r in range(rows):
            for c in range(cols):
                x0, y0 = int(round(c * sx)), int(round(r * sy))
                x1, y1 = min(W, int(round(x0 + tw))), min(H, int(round(y0 + th)))
                tile = frame[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                up = cv2.resize(tile, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc != 1.0 else tile
                try:
                    found = DeepFace.represent(
                        img_path=up, model_name=self.model_name,
                        detector_backend=detector, enforce_detection=True,
                    ) or []
                except Exception:
                    found = []            # no face in this tile
                tile_w, tile_h = x1 - x0, y1 - y0
                for f in found:
                    fa = f.get("facial_area", {})
                    x, y = fa.get("x", 0) / sc, fa.get("y", 0) / sc
                    w, h = fa.get("w", 0) / sc, fa.get("h", 0) / sc
                    # a box touching an INNER tile edge is cut off; the neighbouring tile sees it whole
                    if ((x0 > 0 and x < 3) or (y0 > 0 and y < 3) or
                            (x1 < W and x + w > tile_w - 3) or (y1 < H and y + h > tile_h - 3)):
                        continue
                    new_fa = {"x": int(x0 + x), "y": int(y0 + y), "w": int(w), "h": int(h)}
                    for eye in ("left_eye", "right_eye"):
                        pt = fa.get(eye)
                        if pt is not None:
                            new_fa[eye] = (x0 + pt[0] / sc, y0 + pt[1] / sc)
                    g = dict(f)
                    g["facial_area"] = new_fa
                    out.append(g)
        return out

    @staticmethod
    def _merge_detections(base: List[dict], extra: List[dict], frame_shape: tuple) -> List[dict]:
        """Union of full-frame and tile detections without duplicates (higher
        detector confidence wins). Whole-frame 'fallback' boxes are discarded."""
        fh, fw = frame_shape[:2]
        limit = face_settings.MAX_FACE_AREA_RATIO
        base = [f for f in base
                if (f["facial_area"].get("w", 0) * f["facial_area"].get("h", 0)) / max(fw * fh, 1) <= limit]
        pool = sorted(base + extra, key=lambda f: (f.get("face_confidence") or 1.0), reverse=True)
        kept: List[dict] = []
        for f in pool:
            if not any(_overlaps(f["facial_area"], k["facial_area"]) for k in kept):
                kept.append(f)
        return kept

    def detect_and_embed_faces(
        self, frame: np.ndarray, detector: str = None, rejected_out: Optional[list] = None,
        tiles: Optional[Tuple[int, int]] = None,
    ) -> List[dict]:
        """
        Single detection+embedding path shared by every recognition source
        so webcam, CCTV, and photo upload produce identical results.

        Quality gate is now TIERED (see services/quality_service.py):
          reject → dropped (tiny / unusable / extreme profile)
          low    → kept, but matched with a stricter threshold
          high   → normal matching
        Each returned face carries its "quality" info and a "crop" for review.

        If `rejected_out` (a list) is given, faces dropped by the gate are
        appended to it as {embedding, facial_area, reason, crop, info} so the
        caller can send them to the teacher review page.
        """
        from deepface import DeepFace
        detector = detector or self.detector
        chain = [detector]
        if detector not in ("mtcnn", "opencv"):
            chain += ["mtcnn", "opencv"]
        elif detector != "opencv":
            chain += ["opencv"]

        faces, used_detector = [], None
        for enforce in (True, False):
            for det in chain:
                try:
                    faces = DeepFace.represent(
                        img_path=frame, model_name=self.model_name,
                        detector_backend=det, enforce_detection=enforce,
                    )
                    if faces:
                        used_detector = det
                        break
                except Exception:
                    continue
            if faces:
                break

        if tiles and (tiles[0] > 1 or tiles[1] > 1):
            tiled = self._detect_tiled(frame, detector, tiles)
            before = len(faces or [])
            faces = self._merge_detections(faces or [], tiled, frame.shape)
            print(f"🔲 tiled scan {tiles[0]}x{tiles[1]}: {before} full-frame + {len(tiled)} tile hit(s) → {len(faces)} unique face(s)")
            if faces and used_detector is None:
                used_detector = detector

        if not faces:
            print("👁️  0 face(s) detected")
            return []

        valid = []
        for face_data in faces:
            facial_area = face_data.get("facial_area", {})
            ok, reason = _facial_area_is_valid(facial_area, frame.shape)
            if not ok:
                print(f"  ⏭️  Rejected detection ({reason})")
                if rejected_out is not None and reason.startswith("too_small"):
                    self._collect_rejected(rejected_out, frame, face_data, facial_area, "too_small")
                continue

            confidence = face_data.get("face_confidence")
            if confidence is not None and confidence < face_settings.MIN_DETECTION_CONFIDENCE:
                print(f"  ⏭️  Rejected low-confidence detection ({confidence:.2f})")
                if rejected_out is not None:
                    self._collect_rejected(rejected_out, frame, face_data, facial_area, "low_det_conf")
                continue

            roi = _crop_from_facial_area(frame, facial_area)
            if roi is None or roi.size == 0:
                print("  ⏭️  Rejected detection (empty crop)")
                continue

            yaw = yaw_ratio(
                facial_area.get("left_eye"),
                facial_area.get("right_eye"),
                box=(
                    facial_area.get("x", 0), facial_area.get("y", 0),
                    facial_area.get("w", 0), facial_area.get("h", 0),
                ),
            )
            # Detector confidence is already gated above (MIN_DETECTION_CONFIDENCE),
            # so pass 1.0 here to avoid gating it twice.
            info = quality_info(roi, det_conf=1.0, yaw=yaw)
            print(
                f"  📊 quality={info['tier']} blur={info['blur']} "
                f"width={info['width']}px yaw={info['yaw']}"
            )
            info = _with_cell(info, frame.shape, facial_area)
            if info["tier"] == "reject":
                reason = _reject_reason(info)
                print(f"  ⏭️  Rejected (unusable quality: {reason})")
                if rejected_out is not None:
                    rejected_out.append({
                        "embedding": np.array(face_data["embedding"]),
                        "facial_area": facial_area,
                        "reason": reason,
                        "crop": _crop_with_margin(frame, facial_area),
                        "info": info,
                    })
                continue

            valid.append({
                "embedding": np.array(face_data["embedding"]),
                "facial_area": facial_area,
                "confidence": confidence,
                "quality": info,
                "crop": _crop_with_margin(frame, facial_area),
            })

        print(f"👁️  {len(valid)}/{len(faces)} face(s) passed quality gate (detector={used_detector})")
        return valid

    # ── Cosine matcher (verification step + fallback matcher) ──────────────

    def match_embedding_to_students(
        self,
        query_emb: np.ndarray,
        known_embeddings: List[dict],
        threshold: float = None,
        margin: float = None,
    ) -> Tuple[Optional[dict], float, bool]:
        """
        Returns (best_match_or_None, best_distance, is_confident_match).

        Scores each student by the mean of their top-K closest stored
        (augmented) embeddings, and only accepts a match if it clears
        `threshold` AND beats the runner-up student by at least `margin`.
        """
        threshold = self.threshold if threshold is None else threshold
        margin = self.margin if margin is None else margin

        scored = []
        for known in known_embeddings:
            dists = [
                self._cosine_distance(query_emb, stored_emb)
                for stored_emb in known["embeddings"]
                if len(stored_emb) == len(query_emb)
            ]
            if not dists:
                continue
            dists.sort()
            top_k = dists[: self.top_k]
            scored.append((sum(top_k) / len(top_k), known))

        if not scored:
            return None, float("inf"), False

        scored.sort(key=lambda t: t[0])
        best_distance, best_match = scored[0]

        if best_distance > threshold:
            return None, best_distance, False

        if len(scored) > 1:
            second_distance = scored[1][0]
            if (second_distance - best_distance) < margin:
                print(
                    f"  ⚠️  Ambiguous: {best_match['student_name']} ({best_distance:.4f}) "
                    f"vs runner-up ({second_distance:.4f}) — margin too small, rejecting"
                )
                return None, best_distance, False

        return best_match, best_distance, True

    def match_embedding_to_specific_student(
        self, query_emb: np.ndarray, known: dict, threshold: float = None,
    ) -> Tuple[float, bool]:
        """Cosine distance between query_emb and one specific student's stored
        embeddings — used to verify an SVM prediction (does the top SVM
        class actually look like this face too?)."""
        threshold = self.threshold if threshold is None else threshold
        dists = [
            self._cosine_distance(query_emb, stored_emb)
            for stored_emb in known["embeddings"]
            if len(stored_emb) == len(query_emb)
        ]
        if not dists:
            return float("inf"), False
        dists.sort()
        top_k = dists[: self.top_k]
        mean_dist = sum(top_k) / len(top_k)
        return mean_dist, mean_dist <= threshold

    def _rank_students(self, query_emb: np.ndarray, known_embeddings: List[dict]) -> List[Tuple[float, dict]]:
        """All enrolled students sorted by top-K mean cosine distance (same
        scoring as match_embedding_to_students), IGNORING threshold/margin."""
        q = np.asarray(query_emb, dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        ranked = []
        for known in known_embeddings:
            rows = [e for e in known["embeddings"] if len(e) == len(q)]
            if not rows:
                continue
            M = np.asarray(rows, dtype=np.float64)
            norms = np.linalg.norm(M, axis=1)
            norms[norms == 0] = 1.0
            d = np.sort(1.0 - (M / norms[:, None]) @ q)
            k = min(self.top_k, len(d))
            ranked.append((float(d[:k].mean()), known))
        ranked.sort(key=lambda t: t[0])
        return ranked

    def _best_candidate(
        self, query_emb: np.ndarray, known_embeddings: List[dict],
    ) -> Tuple[Optional[dict], float]:
        """Closest enrolled student ignoring threshold/margin (kept for other callers)."""
        ranked = self._rank_students(query_emb, known_embeddings)
        if not ranked:
            return None, float("inf")
        return ranked[0][1], ranked[0][0]

    @staticmethod
    def _is_marked(session_id, student_id) -> bool:
        if not session_id:
            return False
        try:
            from services.session_manager import session_manager
            # present already, OR set to absent by the teacher (must not be flipped back by the camera)
            return (session_manager.already_recognized(session_id, student_id)
                    or session_manager.is_absent_locked(session_id, student_id))
        except Exception:
            return False

    def _triage(self, query_emb, known_embeddings, session_id, thr, svm_pick=None):
        """For a face that could not be auto-confirmed, decide what the teacher
        should see.

        Returns (drop, candidates, ambiguous, marked_near)
          drop       student dict -> the face is (almost certainly) a student who is
                              ALREADY marked present: no card, only an alert
          marked_near already-marked students that look similar (shown as a note)
          candidates list of {"student_id","student_name","distance"} for students
                     NOT yet marked, best first
          ambiguous  True  -> two or more students are about equally likely
        """
        ranked = self._rank_students(query_emb, known_embeddings)
        if not ranked:
            return None, [], False, []

        top_d, top_k = ranked[0]
        top_marked = self._is_marked(session_id, top_k["student_id"])
        remaining = [(d, k) for d, k in ranked if not self._is_marked(session_id, k["student_id"])]

        ambiguous = False
        if len(ranked) > 1 and top_d <= thr and (ranked[1][0] - top_d) < self.margin:
            ambiguous = True

        if top_marked and top_d <= thr:
            # Looks like a student who is already present. Only keep the face if an
            # UNMARKED student is nearly as close (then it could be them instead).
            close_unmarked = [(d, k) for d, k in remaining if (d - top_d) < self.margin]
            if not close_unmarked:
                return top_k, [], False, []

        cands = [{"student_id": k["student_id"], "student_name": k["student_name"],
                  "distance": d, "via": "cosine"} for d, k in remaining[:REVIEW_TOP_CANDIDATES]]
        # what the SVM thinks goes on the card too (even if cosine rejected it)
        if svm_pick and not self._is_marked(session_id, svm_pick["student"]["student_id"]):
            sid = svm_pick["student"]["student_id"]
            hit = next((c for c in cands if c["student_id"] == sid), None)
            if hit:
                # "both" only if the face is genuinely close; otherwise it is just the SVM's pick
                hit["via"] = "both" if hit["distance"] <= REVIEW_CLOSE_DIST else "svm"
                hit["svm_p"] = round(float(svm_pick["p"]), 3)
            else:
                cands.insert(0, {"student_id": sid, "student_name": svm_pick["student"]["student_name"],
                                 "distance": svm_pick["dist"], "via": "svm",
                                 "svm_p": round(float(svm_pick["p"]), 3)})
                cands = cands[:REVIEW_TOP_CANDIDATES]
        marked_near = [{"student_id": k["student_id"], "student_name": k["student_name"]}
                       for d, k in ranked[:REVIEW_TOP_CANDIDATES]
                       if d <= thr + 0.05 and self._is_marked(session_id, k["student_id"])]
        return None, cands, ambiguous, marked_near

    # ── Recognise ─────────────────────────────────────────────────────────

    def recognize_face_in_frame(
        self,
        frame: np.ndarray,
        known_embeddings: List[dict],
        class_id: str = None,
        check_liveness: bool = True,
        is_cctv: bool = False,
        session_id: str = None,
        source: str = None,
        threshold_delta: float = None,
    ) -> Tuple[List[Tuple[str, str, str, float, bool]], int]:
        """
        Returns (matches, unmatched_count).
          matches: list of (student_id, student_name, email, confidence, is_live)
          unmatched_count: good-quality faces that matched nobody.

        Side effects (the new part):
          * every detected face that is NOT auto-confirmed is sent to the teacher
            review dashboard (quality-rejected, low quality, unmatched, ambiguous
            between students, liveness-failed), with up to 3 candidate students;
          * a face that is confirmed is NOT sent to review;
          * a face that belongs to a student already marked present in this
            session is ignored (not matched again, not sent to review).
        self.last_stats = {"faces", "matched", "sent_to_review", "already_marked"}
        is filled for callers that want to report it.

        `session_id` is needed to know who is already marked; `source` is
        "webcam" | "cctv" | "photo" (defaults from is_cctv); `threshold_delta`
        overrides the per-source threshold allowance (photo passes its own).
        """
        stats = {"faces": 0, "matched": 0, "sent_to_review": 0, "already_marked": 0}
        self.last_stats = stats
        source = source or ("cctv" if is_cctv else "webcam")

        if not known_embeddings:
            print("⚠️  No embeddings — cannot recognise")
            return [], 0

        if is_cctv:
            frame = self.preprocess_cctv_frame(frame)

        detector = "retinaface" if is_cctv else self.detector
        if threshold_delta is None:
            threshold_delta = face_settings.CCTV_THRESHOLD_DELTA if is_cctv else 0.0
        threshold = self.threshold + threshold_delta

        def send(crop, cands, info, reason, emb, mnear=None):
            stats["sent_to_review"] += 1
            _emit_review(class_id, session_id, crop, cands, None, info, reason,
                         embedding=emb, source=source, marked_near=mnear)

        rejected: list = []
        faces = self.detect_and_embed_faces(
            frame, detector=detector, rejected_out=rejected,
            tiles=CCTV_TILES if is_cctv else None,
        )
        stats["faces"] = len(faces) + len(rejected)

        # 1) faces the quality gate dropped -> teacher review (with closest names)
        for rej in rejected:
            drop, cands, _amb, mnear = self._triage(rej["embedding"], known_embeddings, session_id, threshold)
            if drop:
                stats["already_marked"] += 1
                _emit_already_marked(class_id, session_id, drop, source)
                continue
            send(rej["crop"], cands, rej["info"], rej["reason"], rej["embedding"], mnear)

        # Lazy import avoids a circular import (classifier_service uses this module too)
        from services.classifier_service import predict_with_classifier

        by_student_id = {k["student_id"]: k for k in known_embeddings}

        results = []
        unmatched_count = 0
        for face in faces:
            query_emb = face["embedding"]
            facial_area = face["facial_area"]
            info = face.get("quality") or {"tier": "high"}
            tier = info.get("tier", "high")
            crop = face.get("crop")

            # Stricter distance requirement for low-quality faces
            face_threshold = threshold - LOW_QUALITY_THRESHOLD_TIGHTEN if tier == "low" else threshold

            if check_liveness:
                if is_cctv:
                    is_live, liveness_reason, _ = liveness_detector.check_liveness_cctv(frame, facial_area)
                else:
                    is_live, liveness_reason, _ = liveness_detector.check_liveness(frame, facial_area)
                if not is_live:
                    print(f"  🚫 LIVENESS FAILED: {liveness_reason} — sending to teacher review")
                    drop, cands, _amb, mnear = self._triage(query_emb, known_embeddings, session_id, threshold)
                    if drop:
                        stats["already_marked"] += 1
                        _emit_already_marked(class_id, session_id, drop, source)
                    else:
                        send(crop, cands, info, "liveness", query_emb, mnear)
                    continue
            else:
                is_live = True

            best_match, best_distance, confident = None, float("inf"), False
            svm_pick = None        # what the SVM thinks, even when we don't trust it
            disagree = False       # SVM and cosine do not back the same student

            svm_result = predict_with_classifier(class_id, query_emb) if class_id else None
            if svm_result:
                pred_student_id, svm_probability = svm_result
                cand_s = by_student_id.get(pred_student_id)
                if cand_s:
                    cos_dist, cos_ok = self.match_embedding_to_specific_student(
                        query_emb, cand_s, threshold=face_threshold,
                    )
                    svm_pick = {"student": cand_s, "p": svm_probability, "dist": cos_dist, "cos_ok": cos_ok}
                    if svm_probability >= face_settings.SVM_MIN_PROBABILITY:
                        if cos_ok:
                            best_match, best_distance, confident = cand_s, cos_dist, True
                            print(
                                f"  🤖 SVM+cosine agree: {cand_s['student_name']} "
                                f"(svm_p={svm_probability:.2f}, cos_dist={cos_dist:.4f}, quality={tier})"
                            )
                        else:
                            disagree = True
                            print(
                                f"  ⚠️  SVM said {cand_s['student_name']} (p={svm_probability:.2f}) "
                                f"but cosine distance {cos_dist:.4f} > threshold — teacher review"
                            )
                    else:
                        print(f"  ❌ SVM confidence too low (p={svm_probability:.2f})")

            if not confident:
                cm, cd, cc = self.match_embedding_to_students(
                    query_emb, known_embeddings, threshold=face_threshold,
                )
                svm_backs_it = bool(svm_pick and cm is not None
                                    and svm_pick["student"]["student_id"] == cm["student_id"])
                if cc and svm_result and REQUIRE_MODEL_AGREEMENT and not svm_backs_it:
                    # cosine is sure, the SVM names someone else: a human decides
                    disagree = True
                    print(f"  ⚠️  cosine says {cm['student_name']} (dist={cd:.4f}) but the SVM does not agree — teacher review")
                else:
                    best_match, best_distance, confident = cm, cd, cc

            if confident:
                if self._is_marked(session_id, best_match["student_id"]):
                    # already present this session: don't consider them again,
                    # just raise an "already marked" alert on the review page
                    stats["already_marked"] += 1
                    _emit_already_marked(class_id, session_id, best_match, source)
                    continue
                confidence = round((1 - best_distance) * 100, 1)
                print(f"  ✅ MATCHED: {best_match['student_name']} ({confidence}%) [quality={tier}]")
                stats["matched"] += 1
                results.append((
                    best_match["student_id"],
                    best_match["student_name"],
                    best_match["email"],
                    confidence,
                    is_live,
                ))
                continue

            # ---- not confirmed: decide what the teacher sees ----------------
            drop, cands, ambiguous, mnear = self._triage(
                query_emb, known_embeddings, session_id, face_threshold, svm_pick=svm_pick)
            if drop:
                stats["already_marked"] += 1
                _emit_already_marked(class_id, session_id, drop, source)
                print(f"  ⏭️  {drop['student_name']} is already marked present — alert only, not recognised again")
                continue

            if disagree:
                reason = "model_disagree"
                print("  🔍 SVM and distance disagree → review: "
                      + ", ".join(f"{c['student_name']} [{c.get('via')}]" for c in cands[:3]))
            elif ambiguous:
                reason = "ambiguous"
                names = ", ".join(c["student_name"] for c in cands[:2])
                print(f"  🔍 AMBIGUOUS between {names} → review")
            elif tier == "low":
                reason = "low_quality"
                print(f"  🔍 LOW-QUALITY face unresolved → review "
                      f"(closest: {cands[0]['student_name'] if cands else 'none'})")
            else:
                reason = "unmatched"
                unmatched_count += 1
                print(f"  ⚠️  UNKNOWN FACE: detected but not matched (dist={best_distance:.4f}) → review")

            send(crop, cands, info, reason, query_emb, mnear)

        if unmatched_count:
            print(f"🚨 ALERT: {unmatched_count} unrecognized face(s) detected — not enrolled in class {class_id}")

        return results, unmatched_count

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1 - dot / (na * nb)

    @staticmethod
    def decode_base64_frame(b64_data: str) -> Optional[np.ndarray]:
        try:
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"⚠️  Frame decode error: {e}")
            return None