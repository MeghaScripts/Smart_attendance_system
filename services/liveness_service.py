"""
Liveness Detection Service — prevents photo/screen spoofing.
Uses two fast checks:
1. Texture analysis (LBP-based) — real faces have micro-texture, photos don't
2. Eye/glasses check using both eye and eye-with-glasses cascades
Tuned to work reliably with AND without glasses.
"""
import cv2
import numpy as np
from typing import Tuple


class LivenessDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # Also load the glasses-aware cascade — standard eye cascade misses eyes behind lenses
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def compute_lbp_texture_score(self, gray_face: np.ndarray) -> float:
        """
        Local Binary Pattern texture score.
        Real faces: high variance (score > threshold)
        Printed photos / screens: low variance (flat texture)
        """
        h, w = gray_face.shape
        if h < 20 or w < 20:
            return 0.0

        lbp = np.zeros_like(gray_face)
        center = gray_face[1:-1, 1:-1]

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = gray_face[1+dy:h-1+dy, 1+dx:w-1+dx]
                lbp[1:-1, 1:-1] += (neighbor >= center).astype(np.uint8)

        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return float(entropy)

    def check_eyes_present(self, face_roi_gray: np.ndarray) -> bool:
        """
        Check if eyes are detectable using BOTH standard and glasses-aware cascades.
        A real face with glasses will pass the glasses cascade even if the standard one fails.
        minNeighbors lowered to 2 so partial eye visibility (glasses rim) still counts.
        """
        # Try standard eye cascade first
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10)
        )
        if len(eyes) >= 1:
            return True
        # Fall back to glasses-aware cascade
        eyes_glasses = self.eye_glasses_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10)
        )
        return len(eyes_glasses) >= 1

    def check_liveness(self, frame: np.ndarray, facial_area: dict) -> Tuple[bool, str, float]:
        """
        Main liveness check for webcam frames.
        Returns: (is_live, reason, confidence_score)

        Decision rules (tuned for webcam use with/without glasses):
        - texture < 1.8 AND no eyes  → definitely a flat photo, block
        - texture < 1.5              → flat even if "eyes" detected, block
        - everything else            → pass (real face)

        We intentionally err toward letting real faces through rather than
        blocking legitimate students because teachers can manually override.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', frame.shape[1])
            h = facial_area.get('h', frame.shape[0])

            x, y = max(0, x), max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            face_roi = gray[y:y2, x:x2]

            if face_roi.size == 0:
                return True, "no_roi", 50.0  # fallback allow

            texture_score = self.compute_lbp_texture_score(face_roi)
            has_eyes = self.check_eyes_present(face_roi)

            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            is_sharp = laplacian_var > 30  # relaxed from 50 — webcams vary

            print(f"  🔍 Liveness: texture={texture_score:.2f}, eyes={has_eyes}, blur={laplacian_var:.1f}")

            # Hard block: truly flat texture AND no eyes at all
            if texture_score < 1.8 and not has_eyes:
                return False, "flat_texture_no_eyes", texture_score

            # Hard block: extremely flat regardless of eyes
            if texture_score < 1.5:
                return False, "flat_texture", texture_score

            # Soft block: blurry AND flat (printed photo held to cam)
            if not is_sharp and texture_score < 2.0:
                return False, "blurry_flat", texture_score

            liveness_score = min(100, (texture_score / 6.0) * 100)
            return True, "live", round(liveness_score, 1)

        except Exception as e:
            print(f"  ⚠️ Liveness check error: {e}")
            return True, "error_fallback", 50.0  # fail open

    def check_liveness_cctv(self, frame: np.ndarray, facial_area: dict) -> Tuple[bool, str, float]:
        """
        Liveness check for CCTV / top-mounted cameras.
        Skips eye detection entirely — top-down angle means eyes are rarely visible.
        Only uses texture score + blur to reject obvious photo spoofs.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x = max(0, facial_area.get('x', 0))
            y = max(0, facial_area.get('y', 0))
            w = facial_area.get('w', frame.shape[1])
            h = facial_area.get('h', frame.shape[0])
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            face_roi = gray[y:y2, x:x2]

            if face_roi.size == 0:
                return True, "no_roi", 50.0

            texture_score = self.compute_lbp_texture_score(face_roi)
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            is_blurry = laplacian_var < 20  # very relaxed for CCTV compression

            print(f"  🔍 CCTV Liveness: texture={texture_score:.2f}, blur={laplacian_var:.1f}")

            # Only block obvious flat photo spoofs
            if texture_score < 1.5 and is_blurry:
                return False, "flat_texture_blurry", texture_score
            if texture_score < 1.2:
                return False, "flat_texture", texture_score

            liveness_score = min(100, (texture_score / 6.0) * 100)
            return True, "live", round(liveness_score, 1)

        except Exception as e:
            print(f"  ⚠️ CCTV Liveness check error: {e}")
            return True, "error_fallback", 50.0


# Singleton
liveness_detector = LivenessDetector()