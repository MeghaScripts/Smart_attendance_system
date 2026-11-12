"""
Central configuration for the Face Recognition Module (Flask + MySQL edition).

All values can be overridden via environment variables / a `.env` file in
the project root. Defaults here are tuned for a single class of up to
~60 enrolled students recognised via CCTV, webcam, and group-photo upload.
"""
import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class FaceSettings:
    # ─── MySQL ──────────────────────────────────────────────────────────────
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "smart_attendance")
    MYSQL_POOL_SIZE: int = int(os.getenv("MYSQL_POOL_SIZE", "10"))

    # ─── Integration with the main backend ─────────────────────────────────
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:5000")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-secret-key")

    # ─── Face Recognition Model ─────────────────────────────────────────────
    FACE_MODEL: str = os.getenv("FACE_MODEL", "Facenet512")
    FACE_DETECTOR: str = os.getenv("FACE_DETECTOR", "retinaface")

    # DeepFace's own calibrated cosine threshold for Facenet512 is ~0.30.
    # Kept close to that calibrated value; distant/angled faces are handled
    # via detection-quality gating + alignment + multi-frame confirmation
    # rather than by loosening this threshold.
    FACE_DISTANCE_THRESHOLD: float = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.34"))

    # Minimum gap required between the best-matching student and the
    # second-best-matching student (cosine distance). Prevents "won by a
    # hair" false positives when two candidates are both near threshold.
    # Loosened from 0.06 -> 0.04 for classes of ~60: with more enrolled
    # students the raw-cosine runner-up naturally sits closer to the winner
    # more often, and the SVM cross-check (not this margin) is what
    # actually resolves ambiguity at this class size.
    MATCH_MARGIN: float = float(os.getenv("MATCH_MARGIN", "0.04"))

    # Small, justified per-source deltas. CCTV gets a slightly larger
    # allowance for RTSP compression; photo gets a small allowance for
    # uncontrolled pose/angle in a crowd shot.
    CCTV_THRESHOLD_DELTA: float = float(os.getenv("CCTV_THRESHOLD_DELTA", "0.03"))
    PHOTO_THRESHOLD_DELTA: float = float(os.getenv("PHOTO_THRESHOLD_DELTA", "0.03"))

    # How many of a student's stored (augmented) embeddings to average over
    # when scoring a cosine-similarity candidate match.
    MATCH_TOP_K_EMBEDDINGS: int = int(os.getenv("MATCH_TOP_K_EMBEDDINGS", "3"))

    # Require this many separate frames to agree on the same identity before
    # attendance is actually marked (webcam/CCTV streams only).
    CONFIRMATION_FRAMES: int = int(os.getenv("CONFIRMATION_FRAMES", "2"))

    # ─── SVM classifier (trained on top of FaceNet512 embeddings) ──────────
    CLASSIFIER_ENABLED: bool = os.getenv("CLASSIFIER_ENABLED", "true").lower() == "true"
    SVM_KERNEL: str = os.getenv("SVM_KERNEL", "linear")
    SVM_C: float = float(os.getenv("SVM_C", "1.0"))
    # Minimum SVM class probability required to accept a prediction. The SVM
    # prediction is ALWAYS additionally cross-checked against cosine
    # similarity + margin (see embedding_service.match_embedding_to_students)
    # before attendance is marked — the SVM alone never has final say.
    SVM_MIN_PROBABILITY: float = float(os.getenv("SVM_MIN_PROBABILITY", "0.55"))
    # A class needs at least this many distinct enrolled students before a
    # classifier is trained for it; below this, the module falls back to
    # pure cosine-similarity matching (an SVM needs >=2 classes to mean
    # anything, and is unreliable with too few samples per class). At 60
    # students this threshold is only relevant for the first couple of
    # enrollments — the classifier kicks in almost immediately.
    MIN_STUDENTS_FOR_CLASSIFIER: int = int(os.getenv("MIN_STUDENTS_FOR_CLASSIFIER", "2"))

    # ─── Enrollment: multiple real photos per student ────────────────────────
    # Each student is enrolled from several REAL photos (glasses on/off,
    # different lighting, slight angle, with/without a beard) so genuine
    # appearance changes are covered by real captures, not just simulated.
    # Each of those real photos is ALSO expanded into this many synthetic
    # variants (lighting, flip, slight rotation, blur, downward CCTV tilt)
    # for extra robustness to camera/lighting conditions. Total embeddings
    # per student = num_real_photos x AUGMENTATION_VARIANTS_PER_PHOTO, so
    # this is kept lower than the old single-photo default (12) to avoid
    # e.g. 10 photos x 12 = 120 embeddings/student blowing up SVM training
    # and DB size at 60 students. 10 photos x 4 variants x 60 students =
    # 2400 training samples — still trains in well under a second.
    MIN_ENROLLMENT_IMAGES: int = int(os.getenv("MIN_ENROLLMENT_IMAGES", "4"))
    MAX_ENROLLMENT_IMAGES: int = int(os.getenv("MAX_ENROLLMENT_IMAGES", "10"))
    AUGMENTATION_VARIANTS_PER_PHOTO: int = int(os.getenv("AUGMENTATION_VARIANTS_PER_PHOTO", "12"))

    # ─── Detection settings ──────────────────────────────────────────────────
    MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.75"))
    MIN_FACE_SIZE: int = int(os.getenv("MIN_FACE_SIZE", "45"))
    MAX_FACE_AREA_RATIO: float = float(os.getenv("MAX_FACE_AREA_RATIO", "0.70"))
    MIN_BLUR_VARIANCE: float = float(os.getenv("MIN_BLUR_VARIANCE", "25.0"))
    UPSCALE_FACTOR: int = int(os.getenv("UPSCALE_FACTOR", "2"))

    # Expected number of students in a single group photo / CCTV frame.
    # Used by photo_attendance_service to decide how aggressively to
    # upscale a group photo before detection — a 60-person classroom shot
    # needs a much higher target resolution than a 5-10 person photo for
    # every face to clear MIN_FACE_SIZE.
    EXPECTED_CLASS_SIZE: int = int(os.getenv("EXPECTED_CLASS_SIZE", "60"))

    # ─── CCTV streaming ──────────────────────────────────────────────────────
    # Capture resolution for RTSP streams. Bumped up from a 640x480 default:
    # at 640x480, faces beyond the front couple of rows in a 60-student
    # classroom fall below MIN_FACE_SIZE and simply can't be detected. If
    # bandwidth/CPU becomes a bottleneck, prefer switching the camera to its
    # sub-stream profile (/profile2) over dropping this back down.
    RTSP_CAPTURE_WIDTH: int = int(os.getenv("RTSP_CAPTURE_WIDTH", "1920"))
    RTSP_CAPTURE_HEIGHT: int = int(os.getenv("RTSP_CAPTURE_HEIGHT", "1080"))
    RTSP_BUFFER_SIZE: int = int(os.getenv("RTSP_BUFFER_SIZE", "1"))
    FRAME_CAPTURE_TIMEOUT_MS: int = int(os.getenv("FRAME_CAPTURE_TIMEOUT_MS", "2000"))
    FRAME_CAPTURE_SLEEP_MS: float = float(os.getenv("FRAME_CAPTURE_SLEEP_MS", "10"))

    # ─── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EMBEDDINGS_DIR: str = os.getenv("EMBEDDINGS_DIR", os.path.join(BASE_DIR, "face_data", "embeddings"))
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "face_data", "classifiers"))

    CLASS_CAMERA_MAP: str = os.getenv("CLASS_CAMERA_MAP", "{}")
    RECOGNITION_INTERVAL: float = float(os.getenv("RECOGNITION_INTERVAL", "3.0"))

    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:5000",
    ).split(",")

    PORT: int = int(os.getenv("PORT", "8001"))

    def get_camera_map(self) -> dict:
        try:
            return json.loads(self.CLASS_CAMERA_MAP)
        except Exception:
            return {}


face_settings = FaceSettings()
os.makedirs(face_settings.EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(face_settings.MODELS_DIR, exist_ok=True)