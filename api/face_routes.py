import io

import cv2
import requests
from flask import Blueprint, Response, jsonify, request

import db
from core.config import face_settings
from services.session_manager import session_manager
from services.embedding_service import FaceEmbeddingService
from services.classifier_service import train_classifier_for_class, get_classifier_status, invalidate_classifier_cache
from services.cctv_service import (
    cctv_manager, fetch_http_snapshot, probe_http_snapshot,
    cancel_rtsp_test, try_rtsp_connections,
    SNAPSHOT_TEMPLATES, _try_url_basic, _try_url_digest, _try_url_cookie, _decode_frame,
)
from services.photo_attendance_service import PhotoAttendanceService
from services.review_service import drop_if_auto_marked

face_bp = Blueprint("face", __name__, url_prefix="/face")


def _report_attendance(session_id, class_id, student_id, status, source, confidence, svm_probability=None):
    """POST to the main backend (source of truth) and mirror to a local
    audit log. Returns True if the backend confirmed the mark."""
    ok = False
    try:
        resp = requests.post(
            f"{face_settings.BACKEND_URL}/internal/attendance/mark",
            json={
                "session_id": session_id,
                "student_id": str(student_id),
                "status": status,
                "source": source,
                "confidence": confidence,
            },
            timeout=10,
        )
        ok = resp.status_code == 200
    except Exception as e:
        print(f"  ⚠️ Backend mark failed: {e}")

    try:
        db.execute(
            """
            INSERT INTO attendance_log
                (session_id, class_id, student_id, source, confidence, svm_probability, backend_synced)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (session_id, class_id, student_id, source, confidence, svm_probability, ok),
        )
    except Exception as e:
        print(f"  ⚠️ Local attendance log failed: {e}")

    return ok


def _frame_response(service, matches, unmatched_count, reported):
    """Common JSON for webcam / CCTV frame endpoints."""
    stats = getattr(service, "last_stats", {}) or {}
    n_review = stats.get("sent_to_review", 0)
    response = {
        "faces_detected": stats.get("faces", len(matches) + unmatched_count),
        "new_recognitions": reported,
        "unknown_faces_detected": unmatched_count,
        "sent_to_review": n_review,
    }
    if n_review:
        response["alert"] = f"{n_review} face(s) sent to the Review dashboard (/review) for the teacher."
    return response


# ─── Enroll ──────────────────────────────────────────────────────────────────

@face_bp.route("/enroll", methods=["POST"])
def enroll_face():
    student_id = request.form.get("student_id")
    photos = request.files.getlist("photo")  # multiple files under the same "photo" field

    if not student_id or not photos:
        return jsonify({
            "detail": (
                f"student_id and at least {face_settings.MIN_ENROLLMENT_IMAGES} photo(s) "
                f"are required (send multiple files under the 'photo' field, max "
                f"{face_settings.MAX_ENROLLMENT_IMAGES})."
            )
        }), 400

    image_bytes_list = [p.read() for p in photos]
    service = FaceEmbeddingService()
    try:
        result = service.enroll_student(student_id, image_bytes_list)
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400
    except Exception as e:
        return jsonify({"detail": f"Enrollment failed: {e}"}), 500

    # Enrollment was blocked (face too similar to another already-enrolled
    # student in the same class) — nothing was written to the DB, so skip
    # cache invalidation / retraining and report the conflict.
    if result.get("blocked"):
        return jsonify({"success": False, **result}), 409

    session_manager.clear_all_embedding_caches()
    session_manager.reload_embeddings_for_active_sessions()

    # Retrain the class's SVM in the background thread pool of Flask's
    # dev server would block the request — keep it synchronous but fast:
    # training on up to ~60 students takes well under a second.
    student = db.fetch_one("SELECT class_id FROM students WHERE student_id = %s", (student_id,))
    train_result = {"trained": False, "reason": "student_not_found"}
    if student:
        class_id = student["class_id"]
        invalidate_classifier_cache(class_id)
        train_result = train_classifier_for_class(class_id)

    return jsonify({"success": True, **result, "classifier": train_result})


# ─── Classifier training (manual trigger / status) ─────────────────────────

@face_bp.route("/train/<class_id>", methods=["POST"])
def train_classifier(class_id):
    result = train_classifier_for_class(class_id)
    return jsonify(result)


@face_bp.route("/train/<class_id>/status", methods=["GET"])
def classifier_status(class_id):
    return jsonify(get_classifier_status(class_id))


# ─── Session Control ─────────────────────────────────────────────────────────

@face_bp.route("/session/start", methods=["POST"])
def start_recognition_session():
    body = request.get_json(force=True)
    session_id = body["session_id"]
    class_id = body["class_id"]
    teacher_id = body["teacher_id"]
    use_cctv = body.get("use_cctv", False)
    rtsp_url = body.get("rtsp_url")
    nvr_ip = body.get("nvr_ip")
    nvr_http_port = body.get("nvr_http_port", 9008)
    nvr_rtsp_port = body.get("nvr_rtsp_port", 554)
    nvr_channel = body.get("nvr_channel", 1)
    nvr_username = body.get("nvr_username", "admin")
    nvr_password = body.get("nvr_password")
    nvr_brand = body.get("nvr_brand", "tvt")

    if session_manager.is_active(class_id):
        session_manager.end_session(class_id)
        cctv_manager.stop_stream(class_id)

    service = FaceEmbeddingService()
    embeddings = service.get_class_embeddings(class_id)
    session_manager.cache_embeddings(class_id, embeddings)
    session_manager.start_session(class_id, session_id, teacher_id)

    cctv_mode = "none"
    if use_cctv:
        ok, used_url = cctv_manager.start_stream(
            class_id=class_id,
            rtsp_url=rtsp_url or None,
            ip=nvr_ip,
            http_port=nvr_http_port,
            rtsp_port=nvr_rtsp_port,
            channel=nvr_channel,
            username=nvr_username,
            password=nvr_password,
            brand=nvr_brand,
        )
        if ok:
            cctv_mode = used_url
            print(f"📡 CCTV started for {class_id}: {cctv_mode}")
        else:
            print(f"⚠️  CCTV unavailable for {class_id} — no stream")

    return jsonify({
        "success": True,
        "session_id": session_id,
        "class_id": class_id,
        "enrolled_students": len(embeddings),
        "classifier": get_classifier_status(class_id),
        "cctv_active": use_cctv,
        "cctv_mode": cctv_mode,
        "message": f"Recognition active. {len(embeddings)} students enrolled.",
    })


@face_bp.route("/session/end", methods=["POST"])
def end_recognition_session():
    body = request.get_json(force=True)
    class_id = body["class_id"]
    session_manager.end_session(class_id)
    cctv_manager.stop_stream(class_id)
    return jsonify({"success": True, "message": "Recognition session ended"})


# ─── CCTV Live Stream ─────────────────────────────────────────────────────────

@face_bp.route("/cctv/stream/<class_id>", methods=["GET"])
def cctv_mjpeg_stream(class_id):
    if not cctv_manager.is_running(class_id):
        return jsonify({"detail": "No active CCTV stream for this class"}), 404
    return Response(
        cctv_manager.mjpeg_generator(class_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@face_bp.route("/cctv/snapshot/<class_id>", methods=["GET"])
def cctv_snapshot(class_id):
    frame = cctv_manager.get_latest_frame(class_id)
    if frame is None:
        snap_cfg = cctv_manager.get_snapshot_cfg(class_id)
        if snap_cfg:
            frame = fetch_http_snapshot(
                snap_cfg["ip"], snap_cfg["channel"],
                snap_cfg["username"], snap_cfg["password"],
                port=snap_cfg.get("port", 9008),
            )
    if frame is None:
        return jsonify({"detail": "No frame available"}), 404

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@face_bp.route("/cctv/process-now/<class_id>", methods=["POST"])
def cctv_process_now(class_id):
    if not session_manager.is_active(class_id):
        return jsonify({"detail": "No active recognition session"}), 400

    frame = cctv_manager.get_latest_frame(class_id)
    if frame is None:
        snap_cfg = cctv_manager.get_snapshot_cfg(class_id)
        if snap_cfg:
            frame = fetch_http_snapshot(
                snap_cfg["ip"], snap_cfg["channel"],
                snap_cfg["username"], snap_cfg["password"],
                port=snap_cfg.get("port", 9008),
            )
            if frame is not None:
                lock = cctv_manager._locks.get(class_id)
                if lock:
                    with lock:
                        cctv_manager._latest_frames[class_id] = frame.copy()

    if frame is None:
        return jsonify({"error": "No frame from CCTV yet", "faces_detected": 0, "new_recognitions": []})

    session = session_manager.get_session(class_id)
    session_id = session["session_id"]

    print(f"📡 [CCTV-MANUAL] process-now triggered — class={class_id}")
    scan_lock = session_manager.scan_lock(class_id)
    if not scan_lock.acquire(blocking=False):
        return jsonify({"error": "A scan is already running for this class — try again in a few seconds"}), 409
    try:
        service = FaceEmbeddingService()
        embeddings = session_manager.get_cached_embeddings(class_id)
        if not embeddings:
            embeddings = service.get_class_embeddings(class_id)
            session_manager.cache_embeddings(class_id, embeddings)
        # same settings as the background CCTV loop (no webcam-style liveness on a ceiling camera)
        matches, unmatched_count = service.recognize_face_in_frame(
            frame, embeddings, class_id=class_id, check_liveness=False, is_cctv=True,
            session_id=session_id, source="cctv",
        )
    finally:
        scan_lock.release()
    reported = []

    for student_id, student_name, email, confidence, *_ in matches:
        if session_manager.already_recognized(session_id, student_id):
            continue
        if not session_manager.register_candidate_match(session_id, student_id):
            continue
        if _report_attendance(session_id, class_id, student_id, "present", "cctv", confidence):
            session_manager.mark_recognized(session_id, student_id, "cctv")
            session_manager.increment_recognition_count(class_id)
            drop_if_auto_marked(session_id, str(student_id))
            reported.append({
                "student_id": student_id,
                "student_name": student_name,
                "email": email,
                "confidence": confidence,
            })
            print(f"✅ CCTV recognized: {student_name} ({confidence}%)")

    session_manager.record_scan(session_id, "cctv", getattr(service, "last_stats", {}))
    return jsonify(_frame_response(service, matches, unmatched_count, reported))


# ─── Webcam Frame ─────────────────────────────────────────────────────────────

@face_bp.route("/process-frame", methods=["POST"])
def process_frame():
    body = request.get_json(force=True)
    class_id = body["class_id"]
    session_id = body["session_id"]
    frame_b64 = body["frame_b64"]

    if not session_manager.is_active(class_id):
        return jsonify({
            "detail": "No active recognition session for this class (the server may have been restarted) — start the session again",
            "code": "session_not_active",
        }), 400

    service = FaceEmbeddingService()
    frame = service.decode_base64_frame(frame_b64)
    if frame is None:
        return jsonify({"detail": "Could not decode frame"}), 400

    print(f"🎥 [WEBCAM] frame received — class={class_id}, session={session_id}")
    session_manager.increment_frame_count(class_id)
    embeddings = session_manager.get_cached_embeddings(class_id)
    matches, unmatched_count = service.recognize_face_in_frame(
        frame, embeddings, class_id=class_id, session_id=session_id, source="webcam",
    )
    reported = []

    for student_id, student_name, email, confidence, *_ in matches:
        if session_manager.already_recognized(session_id, student_id):
            continue
        if not session_manager.register_candidate_match(session_id, student_id):
            continue
        if _report_attendance(session_id, class_id, student_id, "present", "face_recognition", confidence):
            session_manager.mark_recognized(session_id, student_id, "webcam")
            session_manager.increment_recognition_count(class_id)
            drop_if_auto_marked(session_id, str(student_id))
            reported.append({
                "student_id": student_id,
                "student_name": student_name,
                "email": email,
                "confidence": confidence,
            })

    session_manager.record_scan(session_id, "webcam", getattr(service, "last_stats", {}))
    response = _frame_response(service, matches, unmatched_count, reported)
    response["session_id"] = session_id
    return jsonify(response)


# ─── Photo Attendance ─────────────────────────────────────────────────────────

@face_bp.route("/photo-attendance", methods=["POST"])
def photo_attendance():
    session_id = request.form.get("session_id")
    class_id = request.form.get("class_id")
    photo = request.files.get("photo")

    if not session_id or not class_id or not photo:
        return jsonify({"detail": "session_id, class_id and photo are required"}), 400

    image_bytes = photo.read()
    service = PhotoAttendanceService()
    try:
        result = service.process_class_photo(image_bytes, class_id, session_id)
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400
    except Exception as e:
        return jsonify({"detail": f"Photo processing failed: {e}"}), 500

    session_manager.record_scan(session_id, "photo", {
        "faces": result.get("total_faces_detected", 0), "matched": len(result["matched"]),
        "sent_to_review": result.get("sent_to_review", 0), "already_marked": result.get("already_marked", 0),
    })
    newly_marked = []
    for m in result["matched"]:
        if session_manager.already_recognized(session_id, m["student_id"]):
            continue                      # already present this session
        if _report_attendance(session_id, class_id, m["student_id"], "present", "photo", m["confidence"]):
            session_manager.mark_recognized(session_id, m["student_id"], "photo")
            session_manager.increment_recognition_count(class_id)
            drop_if_auto_marked(session_id, str(m["student_id"]))
            newly_marked.append(m)
    result["matched"] = newly_marked

    if result.get("sent_to_review"):
        result["alert"] = (
            f"{result['sent_to_review']} face(s) sent to the Review dashboard (/review) "
            f"for the teacher."
        )

    return jsonify({"success": True, **result})


# ─── Status ───────────────────────────────────────────────────────────────────

@face_bp.route("/sessions/active", methods=["GET"])
def get_active_sessions():
    return jsonify(session_manager.get_all_active())


@face_bp.route("/attendance-log", methods=["GET"])
def attendance_log():
    """Read-only view of the local audit log (used by the frontend's Records tab).
    Optional query params: session_id, class_id, limit (default 500, max 2000)."""
    where, params = [], []
    for col in ("session_id", "class_id"):
        val = request.args.get(col)
        if val:
            where.append(f"l.{col} = %s")
            params.append(val)
    try:
        limit = max(1, min(int(request.args.get("limit", 500)), 2000))
    except ValueError:
        limit = 500

    query = (
        "SELECT l.id, l.session_id, l.class_id, l.student_id, s.student_name, "
        "l.source, l.confidence, l.svm_probability, l.backend_synced, l.marked_at "
        "FROM attendance_log l LEFT JOIN students s ON s.student_id = l.student_id "
        + (f"WHERE {' AND '.join(where)} " if where else "")
        + "ORDER BY l.id DESC LIMIT %s"
    )
    try:
        rows = db.fetch_all(query, tuple(params) + (limit,))
    except Exception as e:
        return jsonify({"detail": f"Could not read attendance log: {e}"}), 500
    return jsonify({"count": len(rows), "records": rows})


@face_bp.route("/students/<class_id>/enrolled-count", methods=["GET"])
def get_enrolled_count(class_id):
    service = FaceEmbeddingService()
    embeddings = service.get_class_embeddings(class_id)
    return jsonify({"class_id": class_id, "enrolled_count": len(embeddings)})


# ─── CCTV Test Connection ─────────────────────────────────────────────────────

@face_bp.route("/cctv/test", methods=["POST"])
def test_cctv_connection():
    body = request.get_json(force=True)
    nvr_ip = body["nvr_ip"]
    nvr_http_port = body.get("nvr_http_port", 9008)
    nvr_rtsp_port = body.get("nvr_rtsp_port", 554)
    nvr_channel = body.get("nvr_channel", 1)
    nvr_username = body.get("nvr_username", "admin")
    nvr_password = body["nvr_password"]

    ok, desc = probe_http_snapshot(nvr_ip, nvr_channel, nvr_username, nvr_password, port=nvr_http_port)
    if ok:
        return jsonify({"success": True, "url_used": desc, "method": "http_snapshot"})

    cap, url = try_rtsp_connections(nvr_ip, nvr_rtsp_port, nvr_channel, nvr_username, nvr_password)
    if cap:
        cap.release()
        return jsonify({
            "success": True,
            "url_used": url.replace(nvr_password, "****"),
            "method": "rtsp",
        })

    return jsonify({
        "detail": f"HTTP snapshot failed ({desc}). RTSP also failed. Check IP/password."
    }), 500


@face_bp.route("/cctv/cancel-test", methods=["POST"])
def cancel_cctv_test():
    cancel_rtsp_test()
    return jsonify({"cancelled": True})


# ─── Diagnose: raw HTTP probe ─────────────────────────────────────────────────

@face_bp.route("/cctv/diagnose", methods=["POST"])
def diagnose_cctv():
    body = request.get_json(force=True)
    nvr_ip = body["nvr_ip"]
    nvr_http_port = body.get("nvr_http_port", 9008)
    nvr_channel = body.get("nvr_channel", 1)
    nvr_username = body.get("nvr_username", "admin")
    nvr_password = body["nvr_password"]

    results = []
    for p in [nvr_http_port, 80, 8080]:
        for tmpl in SNAPSHOT_TEMPLATES:
            try:
                url = tmpl.format(ip=nvr_ip, port=p, ch=nvr_channel)
            except KeyError:
                continue
            for fn_name, fn in [("basic", _try_url_basic), ("digest", _try_url_digest)]:
                try:
                    data = fn(url, nvr_username, nvr_password, timeout=3)
                    if data is None:
                        results.append({"url": url, "auth": fn_name, "status": "no_response_or_error"})
                    elif _decode_frame(data) is not None:
                        results.append({"url": url, "auth": fn_name, "status": "✅ VALID JPEG", "bytes": len(data)})
                    else:
                        preview = data[:200].decode("utf-8", errors="replace")
                        results.append({
                            "url": url, "auth": fn_name, "status": "got_data_not_jpeg",
                            "bytes": len(data), "preview": preview,
                        })
                except Exception as e:
                    results.append({"url": url, "auth": fn_name, "status": f"error: {e}"})

    data = _try_url_cookie(nvr_ip, nvr_http_port, nvr_username, nvr_password, nvr_channel)
    if data and _decode_frame(data) is not None:
        results.append({"url": f"cookie_auth port {nvr_http_port}", "auth": "cookie", "status": "✅ VALID JPEG"})
    else:
        results.append({
            "url": f"cookie_auth port {nvr_http_port}", "auth": "cookie",
            "status": "failed" if not data else "got_data_not_jpeg",
        })

    working = [r for r in results if "✅" in r.get("status", "")]
    return jsonify({
        "working": working,
        "all_results": results,
        "summary": f"{len(working)} working method(s) found out of {len(results)} tried",
    })