import os
import threading
import time

import numpy as np
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from core.config import face_settings
import db
from api.face_routes import face_bp, _report_attendance
from services.cctv_service import cctv_manager
from services.session_manager import session_manager
from services.embedding_service import FaceEmbeddingService, set_review_sink
from services.review_service import init_review, queue_candidate, drop_if_auto_marked, note_already_marked
from services.classifier_service import train_classifier_for_class, invalidate_classifier_cache

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


# ── Shared attendance marking (auto-recognition AND teacher review) ────────

def _mark_present(session_id: str, student_id, source: str, confidence=None) -> bool:
    """POST one 'present' mark to the backend. Returns True on HTTP 200."""
    try:
        resp = requests.post(
            f"{face_settings.BACKEND_URL}/internal/attendance/mark",
            json={
                "session_id": session_id,
                "student_id": str(student_id),
                "status": "present",
                "source": source,
                # teacher-confirmed marks have no model confidence
                "confidence": confidence if confidence is not None else 0.0,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"⚠️  Mark request failed for {student_id}: {e}")
        return False


# `source` sent to the backend / attendance_log for teacher-confirmed marks.
# Default is a value your routes already use. If your backend and attendance_log
# accept a new value, set REVIEW_MARK_SOURCE=teacher_review in .env.
REVIEW_MARK_SOURCE = os.getenv("REVIEW_MARK_SOURCE", "face_recognition")


def _mark_from_review(session_id: str, class_id: str, student_id) -> None:
    """Called by the review queue when a teacher confirms / assigns.
    Uses the same _report_attendance() as the webcam/CCTV routes, so the mark
    also appears in the attendance_log (Records tab).
    Raises on failure so the review card stays 'pending' and can be retried."""
    if session_manager.already_recognized(session_id, student_id):
        return  # already present — nothing to do
    if not _report_attendance(session_id, class_id, student_id, "present", REVIEW_MARK_SOURCE, 0.0):
        raise RuntimeError("backend did not confirm the mark (is the backend running?)")
    session_manager.mark_recognized(session_id, student_id, "review")
    session_manager.increment_recognition_count(class_id)
    print(f"🧑‍🏫 REVIEW ✅ {student_id} marked present by teacher → {class_id}")


def _set_status_from_review(session_id: str, class_id: str, student_id, status: str, reviewer=None):
    """Teacher manually flips a student on the roster (present <-> absent).
    Returns an optional warning string. Raises if a PRESENT mark could not be saved."""
    student_id = str(student_id)
    if status == "present":
        session_manager.clear_absent_lock(session_id, student_id)
        if session_manager.already_recognized(session_id, student_id):
            return None
        if not _report_attendance(session_id, class_id, student_id, "present", REVIEW_MARK_SOURCE, 0.0):
            raise RuntimeError("backend did not confirm the mark (is the backend running?)")
        session_manager.mark_recognized(session_id, student_id, "manual")
        session_manager.increment_recognition_count(class_id)
        print(f"🧑‍🏫 MANUAL ✅ {student_id} set PRESENT by {reviewer or 'teacher'} → {class_id}")
        return None

    # absent: stop the camera from flipping the student back, then tell the backend
    session_manager.unmark_recognized(session_id, student_id)
    session_manager.lock_absent(session_id, student_id)
    try:
        resp = requests.post(
            f"{face_settings.BACKEND_URL}/internal/attendance/mark",
            json={"session_id": session_id, "student_id": student_id, "status": "absent",
                  "source": REVIEW_MARK_SOURCE, "confidence": 0.0},
            timeout=10,
        )
        ok = resp.status_code == 200
    except Exception as e:
        print(f"⚠️  absent request failed for {student_id}: {e}")
        ok = False
    print(f"🧑‍🏫 MANUAL ❌ {student_id} set ABSENT by {reviewer or 'teacher'} → {class_id} (backend ok={ok})")
    return None if ok else "Saved here, but the backend did not confirm the change — check that it accepts status 'absent'."


# ── Review-queue plumbing ──────────────────────────────────────────────────

def _active_session_for_class(class_id):
    """session_id of the active session for this class (None if none)."""
    try:
        sess = session_manager.get_session(class_id)
        if sess and sess.get("session_id"):
            return sess["session_id"]
    except Exception:
        pass
    for s in session_manager.get_all_active():
        if str(s.get("class_id")) == str(class_id) and s.get("session_id"):
            return s["session_id"]
    return None


def _review_sink(p: dict):
    """Called by FaceEmbeddingService for EVERY face it could not confirm
    (quality-rejected, low quality, unmatched, ambiguous, liveness-failed) from
    webcam, CCTV and photo. Recognised faces never come here."""
    class_id = p["class_id"]
    session_id = p.get("session_id") or _active_session_for_class(class_id)
    if not session_id:
        return
    if p.get("kind") == "already_marked":
        note_already_marked(session_id, class_id, p["student_id"], p.get("student_name"), p.get("source"))
        return
    queue_candidate(
        session_id, class_id, p["crop"], p.get("candidates"), p.get("info"),
        p.get("reason", "unmatched"), embedding=p.get("embedding"), source=p.get("source"),
        marked_near=p.get("marked_near"),
    )


# ── Learn from teacher decisions ───────────────────────────────────────────
# When the teacher confirms/assigns a face, that face (a real example of how the
# student looks on THIS camera) is added to the student's stored embeddings, so
# the next blurry/angled sighting is recognised automatically. Turn off with
# REVIEW_LEARN=0.
REVIEW_LEARN = os.getenv("REVIEW_LEARN", "1") == "1"
REVIEW_LEARN_MAX_EMBEDDINGS = int(os.getenv("REVIEW_LEARN_MAX_EMBEDDINGS", 400))
REVIEW_LEARN_SANITY_DIST = float(os.getenv("REVIEW_LEARN_SANITY_DIST", 0.60))   # guards against wrong assignments
REVIEW_LEARN_MIN_NOVELTY = float(os.getenv("REVIEW_LEARN_MIN_NOVELTY", 0.05))   # skip near-duplicates


def _cos_dist_to_set(vec, mat):
    vn = vec / (np.linalg.norm(vec) or 1.0)
    mn = mat / np.where(np.linalg.norm(mat, axis=1, keepdims=True) == 0, 1.0, np.linalg.norm(mat, axis=1, keepdims=True))
    return 1.0 - mn @ vn


def _learn_from_review(class_id, student_id, emb):
    if not REVIEW_LEARN or emb is None:
        return
    row = db.fetch_one("SELECT embeddings FROM face_embeddings WHERE student_id = %s", (student_id,))
    if not row:
        return
    stored = db.from_json(row["embeddings"]) or []
    if not stored or len(stored) >= REVIEW_LEARN_MAX_EMBEDDINGS:
        return
    mat = np.asarray(stored, dtype=np.float64)
    vec = np.asarray(emb, dtype=np.float64)
    if mat.shape[1] != vec.shape[0]:
        return
    d = np.sort(_cos_dist_to_set(vec, mat))
    if float(d[:3].mean()) > REVIEW_LEARN_SANITY_DIST:
        print(f"ℹ️  review-learn skipped for {student_id}: face too unlike enrolled photos (possible mis-assignment)")
        return
    if float(d[0]) < REVIEW_LEARN_MIN_NOVELTY:
        return                                    # we already have an almost identical sample
    # the review queue stores unit-length vectors; match the scale of the stored ones
    scale = float(np.mean(np.linalg.norm(mat, axis=1)))
    new_emb = (vec / (np.linalg.norm(vec) or 1.0) * scale).tolist()
    stored.append(new_emb)
    db.execute(
        "UPDATE face_embeddings SET embeddings = %s, num_embeddings = %s, updated_at = NOW() WHERE student_id = %s",
        (db.to_json(stored), len(stored), student_id),
    )
    session_manager.clear_all_embedding_caches()
    session_manager.reload_embeddings_for_active_sessions()

    def _retrain():
        try:
            invalidate_classifier_cache(class_id)
            train_classifier_for_class(class_id)
        except Exception as e:
            print(f"⚠️  retrain after review-learn failed: {e}")
    threading.Thread(target=_retrain, daemon=True).start()
    print(f"🧠 review-learn: added a verified sample for {student_id} ({len(stored)} embeddings)")


def _students_for_class(class_id):
    rows = db.fetch_all(
        "SELECT student_id, student_name FROM students WHERE class_id = %s ORDER BY student_name",
        (class_id,),
    )
    return [
        {"student_id": str(r["student_id"]), "student_name": r.get("student_name") or str(r["student_id"])}
        for r in rows
    ]


def _active_sessions():
    return [
        {"session_id": s.get("session_id"), "class_id": s.get("class_id")}
        for s in session_manager.get_all_active() if s.get("session_id")
    ]


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": face_settings.ALLOWED_ORIGINS}}, supports_credentials=True)

    db.init_db_pool()

    app.register_blueprint(face_bp)

    # Teacher review queue API: /face/review/pending, /image/<id>, /<id>/decide
    init_review(
        app, _mark_from_review,
        students_fn=_students_for_class,
        is_marked_fn=lambda sid, stu: session_manager.already_recognized(sid, stu),
        sessions_fn=_active_sessions,
        learn_fn=_learn_from_review,
        roster_info_fn=lambda sid, stu: session_manager.roster_info(sid, stu),
        set_status_fn=_set_status_from_review,
        scan_fn=lambda sid: session_manager.scan_info(sid),
    )
    set_review_sink(_review_sink)

    @app.route("/")
    def root():
        return jsonify({"service": "Face Recognition Module", "status": "running"})

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy"})

    # Web console (frontend/index.html), served same-origin so no CORS setup
    # is needed and the browser allows webcam access on http://localhost.
    @app.route("/ui")
    def ui():
        return send_from_directory(_FRONTEND_DIR, "index.html")

    # Teacher review page for blurry / low-confidence faces
    @app.route("/review")
    def review_page():
        return send_from_directory(_FRONTEND_DIR, "review.html")

    return app


# ── Background CCTV recognition loop (threading, not asyncio) ──────────────

def _cctv_recognition_loop():
    """
    Runs every RECOGNITION_INTERVAL seconds in a daemon thread.
    For every active class that has a CCTV stream running, grabs the
    latest frame, runs face recognition (SVM + cosine verification), and
    marks matched students present via the backend's internal API.

    Faces that cannot be confirmed automatically (rejected by the quality
    gate, low quality, unmatched) are NOT dropped: FaceEmbeddingService sends
    them to the teacher review page (/review) through the registered sink.
    """
    time.sleep(5)  # let Flask + DB pool finish starting up
    print("🔄 CCTV recognition loop started")
    service = FaceEmbeddingService()

    while True:
        try:
            for session_info in session_manager.get_all_active():
                class_id = session_info.get("class_id")
                if not class_id or not cctv_manager.is_running(class_id):
                    continue

                frame = cctv_manager.get_latest_frame(class_id)
                if frame is None:
                    continue

                session_id = session_info.get("session_id")
                if not session_id:
                    continue

                scan_lock = session_manager.scan_lock(class_id)
                if not scan_lock.acquire(blocking=False):
                    continue            # a manual "Scan now" is running for this class

                try:
                    print(f"📡 [CCTV-AUTO] background loop processing frame — class={class_id}")
                    embeddings = session_manager.get_cached_embeddings(class_id)
                    if not embeddings:
                        embeddings = service.get_class_embeddings(class_id)
                        session_manager.cache_embeddings(class_id, embeddings)

                    matches, unmatched_count = service.recognize_face_in_frame(
                        frame, embeddings, class_id=class_id,
                        check_liveness=False, is_cctv=True,
                        session_id=session_id, source="cctv",
                    )

                    session_manager.record_scan(session_id, "cctv", getattr(service, "last_stats", {}))

                    if unmatched_count:
                        print(
                            f"🚨 ALERT [CCTV-AUTO]: {unmatched_count} unrecognized face(s) "
                            f"in {class_id} — not enrolled in this class."
                        )

                    for student_id, student_name, email, confidence, _ in matches:
                        if session_manager.already_recognized(session_id, student_id):
                            continue
                        if not session_manager.register_candidate_match(session_id, student_id):
                            continue
                        if _mark_present(session_id, student_id, "cctv", confidence):
                            session_manager.mark_recognized(session_id, student_id, "cctv")
                            session_manager.increment_recognition_count(class_id)
                            drop_if_auto_marked(session_id, str(student_id))
                            print(f"📡 CCTV ✅ {student_name} ({confidence}%) → {class_id}")
                        else:
                            print(f"⚠️  Mark failed for {student_name}")

                except Exception as e:
                    print(f"⚠️  CCTV recognition error for {class_id}: {e}")
                finally:
                    scan_lock.release()

        except Exception as e:
            print(f"⚠️  Recognition loop error: {e}")

        time.sleep(face_settings.RECOGNITION_INTERVAL)


app = create_app()

# Start the background loop once, when this module is imported by
# gunicorn/flask run — guarded so it doesn't double-start under the
# Flask reloader (which imports this module twice in debug mode).
_loop_thread = threading.Thread(target=_cctv_recognition_loop, daemon=True)
_loop_thread.start()

print("🎥 Face Recognition Module ready")
print(f"   Model: {face_settings.FACE_MODEL}")
print(f"   Detector: {face_settings.FACE_DETECTOR}")
print(f"   Threshold: {face_settings.FACE_DISTANCE_THRESHOLD}")
print(f"   Classifier: {'enabled' if face_settings.CLASSIFIER_ENABLED else 'disabled'} "
      f"(SVM, min {face_settings.MIN_STUDENTS_FOR_CLASSIFIER} students/class)")
print("   Review queue: /review (teacher page)")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=face_settings.PORT, debug=False, threaded=True)