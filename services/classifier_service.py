"""
Classifier Service — trains an SVM on top of frozen FaceNet512 embeddings.

Pipeline (per class):
    enrolled students' embeddings  →  StandardScaler  →  SVC(probability=True)
    →  pickled to disk  →  metadata row in MySQL `classifiers` table

FaceNet512 itself is never fine-tuned — only this lightweight classifier is
trained, which is why (re)training takes seconds, not hours, and can be
re-run after every enrollment.

Recognition-time usage lives in embedding_service.recognize_face_in_frame():
the SVM's predicted class is always cross-checked against cosine similarity
to that specific student's stored embeddings before being trusted.
"""
import os
import pickle
import threading
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

import db
from core.config import face_settings

# In-memory cache: class_id -> (scaler, svm, label_encoder)
# Avoids hitting disk on every single frame during a live session.
_classifier_cache = {}
_cache_lock = threading.Lock()


def _model_path(class_id) -> str:
    safe_id = "".join(c for c in str(class_id) if c.isalnum() or c in ("-", "_")) or "unknown"
    return os.path.join(face_settings.MODELS_DIR, f"{safe_id}.pkl")


def train_classifier_for_class(class_id: str) -> dict:
    """
    Trains (or re-trains) the SVM classifier for one class using every
    embedding currently stored for its enrolled students (all augmented
    variants count as separate training samples for that student's class).

    Returns a status dict — never raises for the "not enough data yet" case,
    since that's an expected state early in enrollment (classifier just
    isn't ready; the module falls back to cosine-only matching).
    """
    from services.embedding_service import FaceEmbeddingService

    embedding_service = FaceEmbeddingService()
    students = embedding_service.get_class_embeddings(class_id)

    if len(students) < face_settings.MIN_STUDENTS_FOR_CLASSIFIER:
        print(
            f"ℹ️  {class_id}: only {len(students)} enrolled student(s) — "
            f"need >= {face_settings.MIN_STUDENTS_FOR_CLASSIFIER} to train an SVM. "
            f"Falling back to cosine-only matching."
        )
        return {
            "trained": False,
            "reason": "not_enough_students",
            "num_students": len(students),
            "min_required": face_settings.MIN_STUDENTS_FOR_CLASSIFIER,
        }

    X, y = [], []
    for s in students:
        for emb in s["embeddings"]:
            X.append(emb)
            y.append(s["student_id"])
    X = np.array(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(
        kernel=face_settings.SVM_KERNEL,
        C=face_settings.SVM_C,
        probability=True,
        class_weight="balanced",
    )
    svm.fit(X_scaled, y_encoded)

    path = _model_path(class_id)
    with open(path, "wb") as f:
        pickle.dump({"scaler": scaler, "svm": svm, "label_encoder": label_encoder}, f)

    db.execute(
        """
        INSERT INTO classifiers (class_id, model_path, num_classes, num_samples, trained_at)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            model_path = VALUES(model_path),
            num_classes = VALUES(num_classes),
            num_samples = VALUES(num_samples),
            trained_at = VALUES(trained_at)
        """,
        (class_id, path, len(students), len(X), datetime.utcnow()),
    )

    with _cache_lock:
        _classifier_cache[class_id] = (scaler, svm, label_encoder)

    print(f"✅ SVM trained for {class_id}: {len(students)} students, {len(X)} training samples")
    return {
        "trained": True,
        "num_students": len(students),
        "num_samples": len(X),
        "model_path": path,
    }


def _load_classifier(class_id: str):
    with _cache_lock:
        cached = _classifier_cache.get(class_id)
        if cached:
            return cached

    row = db.fetch_one(
        "SELECT model_path FROM classifiers WHERE class_id = %s", (class_id,)
    )
    if not row or not os.path.exists(row["model_path"]):
        return None

    with open(row["model_path"], "rb") as f:
        payload = pickle.load(f)

    result = (payload["scaler"], payload["svm"], payload["label_encoder"])
    with _cache_lock:
        _classifier_cache[class_id] = result
    return result


def invalidate_classifier_cache(class_id: str = None):
    """Call after re-enrollment so the next prediction re-reads from disk/DB."""
    with _cache_lock:
        if class_id:
            _classifier_cache.pop(class_id, None)
        else:
            _classifier_cache.clear()


def predict_with_classifier(class_id: str, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
    """
    Returns (predicted_student_id, probability) or None if no classifier is
    trained yet for this class. This prediction is NOT final by itself —
    the caller (embedding_service.recognize_face_in_frame) always
    cross-checks it against cosine similarity before accepting it.
    """
    if not face_settings.CLASSIFIER_ENABLED:
        return None

    loaded = _load_classifier(class_id)
    if loaded is None:
        return None
    scaler, svm, label_encoder = loaded

    X = scaler.transform(query_embedding.reshape(1, -1))
    probabilities = svm.predict_proba(X)[0]
    best_idx = int(np.argmax(probabilities))
    best_prob = float(probabilities[best_idx])
    predicted_student_id = label_encoder.inverse_transform([best_idx])[0]

    return predicted_student_id, best_prob


def get_classifier_status(class_id: str) -> dict:
    row = db.fetch_one(
        "SELECT class_id, num_classes, num_samples, trained_at FROM classifiers WHERE class_id = %s",
        (class_id,),
    )
    if not row:
        return {"trained": False, "class_id": class_id}
    return {"trained": True, **row}
