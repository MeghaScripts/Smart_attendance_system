"""
services/quality_service.py

Drop-in module for the Smart Attendance face-recognition service.
Fixes: faces detected on CCTV but rejected by a hard blur gate.

Contains:
  1. Tunables (env-driven, no edit to core/config.py needed)
  2. Soft quality gate: blur + face size + detector confidence + head pose (yaw)
  3. Per-tier match thresholds
  4. degrade(): CCTV-style augmentation (distance blur, motion blur, JPEG) for enrollment
  5. VoteTracker: multi-frame, time-window confirmation before marking attendance

No dependency on the rest of your code. Needs: numpy, opencv-python.
"""

import os
import time
import threading
from collections import defaultdict, deque

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1. Tunables (override in .env). Starting points: tune on YOUR CCTV frames.
# --------------------------------------------------------------------------
MIN_FACE_PX_HARD = int(os.getenv("MIN_FACE_PX_HARD", 40))     # below: reject
MIN_FACE_PX_GOOD = int(os.getenv("MIN_FACE_PX_GOOD", 70))     # below: "low"
BLUR_HARD = float(os.getenv("BLUR_HARD", 15))                 # below: reject
BLUR_GOOD = float(os.getenv("BLUR_GOOD", 60))                 # below: "low"
DET_CONF_MIN = float(os.getenv("DET_CONF_MIN", 0.8))
YAW_GOOD = float(os.getenv("YAW_GOOD", 0.20))                 # near-frontal
YAW_HARD = float(os.getenv("YAW_HARD", 0.45))                 # near-profile: reject

MATCH_THRESHOLD_HIGH = float(os.getenv("MATCH_THRESHOLD_HIGH", 0.37))
MATCH_THRESHOLD_LOW = float(os.getenv("MATCH_THRESHOLD_LOW", 0.30))  # stricter

VOTES_REQUIRED = int(os.getenv("VOTES_REQUIRED", 3))
VOTE_WINDOW_SEC = int(os.getenv("VOTE_WINDOW_SEC", 180))      # 3 min window

DEGRADED_VARIANTS_PER_PHOTO = int(os.getenv("DEGRADED_VARIANTS_PER_PHOTO", 8))


# --------------------------------------------------------------------------
# 2. Quality gate
# --------------------------------------------------------------------------
def blur_score(face_bgr, size=112):
    """Variance of Laplacian on a fixed-size grayscale crop (higher = sharper).
    Resizing to a fixed size makes scores comparable across face sizes."""
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(cv2.resize(face_bgr, (size, size)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def yaw_ratio(left_eye, right_eye, nose=None, box=None):
    """Approximate head turn. 0 = frontal, grows as the head turns.

    Preferred: pass `nose` (x, y). Offset of nose from the eye midpoint,
    divided by eye distance.
    Fallback (DeepFace/RetinaFace output with only eyes): pass `box`
    = (x, y, w, h). Uses offset of the eye midpoint from the box centre.
    Returns 0.0 if it cannot be computed (no penalty).
    """
    try:
        if left_eye is None or right_eye is None:
            return 0.0
        eye_dist = max(1.0, abs(float(right_eye[0]) - float(left_eye[0])))
        mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
        if nose is not None:
            return abs(float(nose[0]) - mid_x) / eye_dist
        if box is not None:
            x, y, w, h = box
            return abs(mid_x - (x + w / 2.0)) / max(1.0, float(w)) * 2.0
    except Exception:
        pass
    return 0.0


def quality_tier(face_bgr, det_conf=1.0, yaw=0.0):
    """Return 'reject', 'low' or 'high'. Also use quality_info() for logging."""
    return quality_info(face_bgr, det_conf, yaw)["tier"]


def quality_info(face_bgr, det_conf=1.0, yaw=0.0):
    """Tier plus the raw numbers (log these while tuning)."""
    w = int(face_bgr.shape[1]) if face_bgr is not None and face_bgr.size else 0
    s = blur_score(face_bgr)
    if (w < MIN_FACE_PX_HARD or det_conf < DET_CONF_MIN
            or s < BLUR_HARD or yaw > YAW_HARD):
        tier = "reject"
    elif s < BLUR_GOOD or w < MIN_FACE_PX_GOOD or yaw > YAW_GOOD:
        tier = "low"
    else:
        tier = "high"
    return {"tier": tier, "blur": round(s, 1), "width": w,
            "yaw": round(float(yaw), 3), "det_conf": round(float(det_conf), 3)}


def threshold_for(tier):
    """Cosine-distance threshold: stricter for low-quality faces."""
    return MATCH_THRESHOLD_HIGH if tier == "high" else MATCH_THRESHOLD_LOW


# --------------------------------------------------------------------------
# 3. Enrollment augmentation: simulate what the CCTV sees
# --------------------------------------------------------------------------
def degrade(img_bgr):
    """One random CCTV-style degradation: distance blur, motion/gaussian blur,
    JPEG compression, slight brightness shift."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # distance: shrink then enlarge
    s = np.random.uniform(0.25, 0.6)
    img = cv2.resize(img, (max(16, int(w * s)), max(16, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # motion or gaussian blur
    if np.random.rand() < 0.5:
        k = int(np.random.choice([5, 7, 9]))
        kern = np.zeros((k, k), dtype=np.float32)
        kern[k // 2, :] = 1.0 / k
        # random direction
        angle = float(np.random.uniform(0, 180))
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kern = cv2.warpAffine(kern, rot, (k, k))
        total = kern.sum()
        if total > 0:
            kern /= total
            img = cv2.filter2D(img, -1, kern)
    else:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)

    # brightness / contrast jitter
    alpha = float(np.random.uniform(0.85, 1.15))
    beta = float(np.random.uniform(-15, 15))
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # JPEG compression artifacts
    q = int(np.random.randint(25, 60))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def make_degraded_variants(img_bgr, n=None):
    """List of n degraded copies. Add these to your existing enrollment variants."""
    n = DEGRADED_VARIANTS_PER_PHOTO if n is None else n
    return [degrade(img_bgr) for _ in range(n)]


# --------------------------------------------------------------------------
# 4. Multi-frame voting inside a time window
# --------------------------------------------------------------------------
class VoteTracker:
    """One instance per recognition session.

    add(student_id) -> True exactly once, when the student reaches
    VOTES_REQUIRED matches within VOTE_WINDOW_SEC. Then send the attendance mark.
    """

    def __init__(self, votes_required=None, window_sec=None):
        self.votes_required = votes_required or VOTES_REQUIRED
        self.window_sec = window_sec or VOTE_WINDOW_SEC
        self._hits = defaultdict(deque)
        self.marked = set()
        self._lock = threading.Lock()

    def add(self, student_id, now=None):
        now = time.time() if now is None else now
        with self._lock:
            if student_id in self.marked:
                return False
            q = self._hits[student_id]
            q.append(now)
            while q and now - q[0] > self.window_sec:
                q.popleft()
            if len(q) >= self.votes_required:
                self.marked.add(student_id)
                return True
            return False

    def count(self, student_id):
        with self._lock:
            return len(self._hits[student_id])

    def reset(self):
        with self._lock:
            self._hits.clear()
            self.marked.clear()


# --------------------------------------------------------------------------
# Integration example (adapt names to your code)
# --------------------------------------------------------------------------
# from services.quality_service import (
#     quality_info, threshold_for, make_degraded_variants, VoteTracker, yaw_ratio)
#
# tracker = VoteTracker()                      # once per session
#
# for face in detected_faces:                  # inside your per-frame loop
#     yaw = yaw_ratio(face["left_eye"], face["right_eye"],
#                     nose=face.get("nose"),
#                     box=(fx, fy, fw, fh))
#     info = quality_info(face_crop, face["confidence"], yaw)
#     print("QUALITY", info)                   # log while tuning
#     if info["tier"] == "reject":
#         continue
#
#     student_id, dist = your_existing_matcher(face_crop)   # SVM + cosine + margin
#     if student_id and dist < threshold_for(info["tier"]):
#         if tracker.add(student_id):
#             mark_attendance(student_id)      # send to backend once
#
# # Enrollment: after your existing 12 variants
# variants += make_degraded_variants(photo_bgr)
