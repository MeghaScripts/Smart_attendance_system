"""
services/review_service.py   (v3: replaces v2)

Teacher review queue for EVERY face the system could not confirm on its own,
from every source (webcam, CCTV, group photo).

A card is created when a face was detected but NOT auto-marked:
  too_blurry / too_small / extreme_angle / low_det_conf -> rejected by the quality gate
  low_quality   -> sharp enough to try, but not confirmed
  unmatched     -> good face, no confident match
  ambiguous     -> model is torn between 2+ students (all of them are listed)
  liveness      -> liveness check failed (possible photo/screen spoof)

A face that IS recognised and marked never reaches this queue. A student who
is already marked present in the session is never suggested again, and cards
that only pointed at a student who got marked meanwhile disappear.

Each card shows the crop, the reason, the source and up to 3 candidate
students. The teacher can Confirm / pick a candidate / Assign anyone / Dismiss.

Dedupe (so the same person seen every 3 s doesn't create 100 cards):
  * same face (embedding cosine distance < REVIEW_MERGE_DIST) -> one card
  * same top candidate in the same seat area                  -> one card
  The best crop is always kept. Faces the teacher dismissed are not re-queued.

Storage: SQLite (review.db) + image files. Old v1/v2 databases are migrated
in place (new columns are added automatically).
"""

import json
import os
import re
import time
import uuid
import sqlite3
import threading

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file, abort

REVIEW_DB = os.getenv("REVIEW_DB", "review.db")
REVIEW_DIR = os.getenv("REVIEW_DIR", "review_images")
REVIEW_MAX_DIST = float(os.getenv("REVIEW_MAX_DIST", 0.55))            # beyond this: not suggested
REVIEW_MAX_CANDIDATES = int(os.getenv("REVIEW_MAX_CANDIDATES", 3))
REVIEW_MERGE_DIST = float(os.getenv("REVIEW_MERGE_DIST", 0.38))        # same-face dedupe (vs ANY stored view of the card)
REVIEW_WEBCAM_MERGE_DIST = float(os.getenv("REVIEW_WEBCAM_MERGE_DIST", 0.55))  # looser: same webcam, same spot, recent
REVIEW_WEBCAM_WINDOW_SEC = float(os.getenv("REVIEW_WEBCAM_WINDOW_SEC", 120))
REVIEW_VIEWS_PER_CARD = int(os.getenv("REVIEW_VIEWS_PER_CARD", 8))     # face embeddings remembered per card
REVIEW_RETENTION_DAYS = int(os.getenv("REVIEW_RETENTION_DAYS", 7))
REVIEW_MAX_PER_SESSION = int(os.getenv("REVIEW_MAX_PER_SESSION", 200))  # pending cards per session

os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(os.path.join(REVIEW_DIR, "confirmed"), exist_ok=True)

review_bp = Blueprint("review", __name__, url_prefix="/face/review")
_lock = threading.RLock()

_mark_fn = None        # (session_id, class_id, student_id) -> None, raises on failure
_students_fn = None    # (class_id) -> [{"student_id","student_name"}, ...]
_is_marked_fn = None   # (session_id, student_id) -> bool
_sessions_fn = None    # () -> [{"session_id","class_id"}, ...]
_learn_fn = None       # (class_id, student_id, embedding) -> None   (optional)
_set_status_fn = None   # (session_id, class_id, student_id, "present"|"absent", reviewer) -> warning|None, raises on failure
_scan_fn = None         # (session_id) -> {source: {faces, matched, sent_to_review, already_marked, at}}
_roster_info_fn = None  # (session_id, student_id) -> {"source","at","confirming","needed"}  (optional)


def _conn():
    c = sqlite3.connect(REVIEW_DB, timeout=15)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS review_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            class_id TEXT NOT NULL,
            face_key TEXT NOT NULL,
            suggested_student_id TEXT,
            distance REAL,
            reason TEXT NOT NULL,
            blur REAL, width INTEGER, yaw REAL, score REAL,
            image_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            final_student_id TEXT,
            reviewer TEXT,
            created_at REAL,
            decided_at REAL,
            UNIQUE(session_id, face_key)
        )""")
        # v3 columns (added in place so an existing review.db keeps working)
        have = {r["name"] for r in c.execute("PRAGMA table_info(review_items)")}
        for col, ddl in (
            ("candidates", "TEXT"),        # JSON [{student_id, distance}]
            ("embedding", "BLOB"),         # float32 bytes of the face embedding
            ("source", "TEXT"),            # webcam | cctv | photo
            ("cell", "TEXT"),              # coarse seat area
            ("seen", "INTEGER DEFAULT 1"),  # how many times this face was seen
            ("last_seen", "REAL"),         # last time this face was seen
            ("marked_near", "TEXT"),       # JSON [{student_id, student_name}] already-marked look-alikes
        ):
            if col not in have:
                c.execute(f"ALTER TABLE review_items ADD COLUMN {col} {ddl}")
        c.execute("""
        CREATE TABLE IF NOT EXISTS review_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, student_id TEXT NOT NULL, status TEXT NOT NULL,
            reviewer TEXT, at REAL, warning TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS review_notices (
            session_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            class_id TEXT,
            student_name TEXT,
            source TEXT,
            count INTEGER DEFAULT 1,
            first_seen REAL,
            last_seen REAL,
            PRIMARY KEY (session_id, student_id)
        )""")


_init_db()


def init_review(app, mark_attendance_fn, students_fn=None, is_marked_fn=None,
                sessions_fn=None, learn_fn=None, roster_info_fn=None, set_status_fn=None, scan_fn=None):
    """Call once in create_app()."""
    global _mark_fn, _students_fn, _is_marked_fn, _sessions_fn, _learn_fn, _roster_info_fn
    global _set_status_fn, _scan_fn
    _mark_fn = mark_attendance_fn
    _students_fn = students_fn
    _is_marked_fn = is_marked_fn
    _sessions_fn = sessions_fn
    _learn_fn = learn_fn
    _roster_info_fn = roster_info_fn
    _set_status_fn = set_status_fn
    _scan_fn = scan_fn
    app.register_blueprint(review_bp)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))


def _score(info):
    # bigger and sharper crop = more useful to the teacher
    return float(info.get("blur", 0) or 0) * min(1.0, float(info.get("width", 0) or 0) / 70.0)


def _emb_to_blob(emb):
    if emb is None:
        return None
    a = np.asarray(emb, dtype=np.float32).ravel()
    n = float(np.linalg.norm(a))
    if n == 0:
        return None
    return (a / n).astype(np.float32).tobytes()          # stored L2-normalised


def _blob_to_emb(blob):
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _views(blob, dim):
    """All embeddings remembered for a card (several views of the same face)."""
    if not blob:
        return None
    a = np.frombuffer(blob, dtype=np.float32)
    if dim and len(a) % dim == 0 and len(a) >= dim:
        return a.reshape(-1, dim)
    return None


def _min_dist(emb, views):
    if views is None or views.shape[1] != len(emb):
        return 9.0
    return float(1.0 - np.max(views @ emb))


def _add_view(blob, emb_blob):
    """Append a new view, keeping the first and the newest REVIEW_VIEWS_PER_CARD-1."""
    new = np.frombuffer(emb_blob, dtype=np.float32)
    dim = len(new)
    v = _views(blob, dim)
    if v is None:
        return emb_blob
    v = np.vstack([v, new])
    if len(v) > REVIEW_VIEWS_PER_CARD:
        v = np.vstack([v[:1], v[-(REVIEW_VIEWS_PER_CARD - 1):]])
    return v.astype(np.float32).tobytes()


def _marked(session_id, student_id):
    try:
        return bool(_is_marked_fn and _is_marked_fn(session_id, str(student_id)))
    except Exception:
        return False


def _remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Called by the recognition code (through the sink in main.py)
# --------------------------------------------------------------------------
def note_already_marked(session_id, class_id, student_id, student_name=None, source=None):
    """A face of a student who is ALREADY marked present was seen again. Not a
    card: it is listed as an alert on the review page (one row per student)."""
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO review_notices (session_id, student_id, class_id, student_name, source, count, "
            "first_seen, last_seen) VALUES (?,?,?,?,?,1,?,?) "
            "ON CONFLICT(session_id, student_id) DO UPDATE SET count=count+1, last_seen=excluded.last_seen, "
            "source=excluded.source",
            (session_id, str(student_id), str(class_id), student_name, source, now, now))


def queue_candidate(session_id, class_id, face_crop_bgr, candidates, info, reason="low_quality",
                    embedding=None, source=None, marked_near=None):
    """Queue one face for review. Returns True if stored/updated.

    candidates: [{"student_id": ..., "distance": float}, ...] best first (may be empty).
    """
    if face_crop_bgr is None or getattr(face_crop_bgr, "size", 0) == 0:
        return False

    info = info or {}
    cands = []
    for cd in (candidates or []):
        sid = cd.get("student_id")
        dist = cd.get("distance")
        via = cd.get("via")
        # the SVM's own pick is always worth showing; cosine picks must be reasonably close
        if sid in (None, "") or dist is None or (dist > REVIEW_MAX_DIST and via not in ("svm", "both")):
            continue
        if _marked(session_id, sid):
            continue                                   # already present: never suggest again
        entry = {"student_id": str(sid), "distance": round(float(dist), 4)}
        if via:
            entry["via"] = via
        if cd.get("svm_p") is not None:
            entry["svm_p"] = cd["svm_p"]
        cands.append(entry)
        if len(cands) >= REVIEW_MAX_CANDIDATES:
            break

    top = cands[0] if cands else None
    score = _score(info)
    cell = info.get("cell", "0_0")
    emb_blob = _emb_to_blob(embedding)
    emb = _blob_to_emb(emb_blob)

    now = time.time()
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT id, score, status, embedding, suggested_student_id, cell, image_path, seen, source, "
            "created_at, last_seen FROM review_items WHERE session_id=? AND status IN ('pending','rejected')",
            (session_id,)).fetchall()

        # ---- find an existing card for the same face ------------------------
        match, best_d = None, 9.0
        if emb is not None:
            for r in rows:
                d = _min_dist(emb, _views(r["embedding"], len(emb)))
                # rule A: looks like ANY remembered view of that card
                if d < REVIEW_MERGE_DIST and d < best_d:
                    match, best_d = r, d
                    continue
                # rule C: same webcam, same spot on screen, seen a moment ago
                last = r["last_seen"] or r["created_at"] or 0
                if (source == "webcam" and r["source"] == "webcam" and r["cell"] == cell
                        and now - last < REVIEW_WEBCAM_WINDOW_SEC
                        and d < REVIEW_WEBCAM_MERGE_DIST and d < best_d):
                    match, best_d = r, d
        if match is None and top is not None:
            for r in rows:
                # rule B: same suggested student in the same seat area
                if (r["status"] == "pending" and r["suggested_student_id"] == top["student_id"]
                        and r["cell"] == cell):
                    match = r
                    break

        if match is not None:
            if match["status"] != "pending":
                return False                           # teacher already dismissed this face
            views = _add_view(match["embedding"], emb_blob) if emb_blob else match["embedding"]
            if (match["score"] or 0) >= score:
                c.execute("UPDATE review_items SET seen=COALESCE(seen,1)+1, embedding=?, last_seen=? WHERE id=?",
                          (views, now, match["id"]))
                return False                           # existing crop is better
            path = match["image_path"]
            cv2.imwrite(path, face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            c.execute(
                "UPDATE review_items SET suggested_student_id=?, distance=?, candidates=?, reason=?, "
                "blur=?, width=?, yaw=?, score=?, embedding=?, source=?, cell=?, created_at=?, last_seen=?, "
                "seen=COALESCE(seen,1)+1 WHERE id=?",
                (top["student_id"] if top else None, top["distance"] if top else None,
                 json.dumps(cands), reason, info.get("blur"), info.get("width"), info.get("yaw"),
                 score, views, source, cell, now, now, match["id"]))
            return True

        # ---- new card --------------------------------------------------------
        n = c.execute("SELECT COUNT(*) FROM review_items WHERE session_id=? AND status='pending'",
                      (session_id,)).fetchone()[0]
        if n >= REVIEW_MAX_PER_SESSION:
            return False

        key = uuid.uuid4().hex
        path = os.path.join(REVIEW_DIR, f"{_safe(session_id)}_{key}.jpg")
        cv2.imwrite(path, face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        c.execute(
            "INSERT INTO review_items (session_id, class_id, face_key, suggested_student_id, distance, "
            "reason, blur, width, yaw, score, image_path, created_at, candidates, embedding, source, cell, seen, "
            "marked_near, last_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
            (session_id, str(class_id), key, top["student_id"] if top else None,
             top["distance"] if top else None, reason, info.get("blur"), info.get("width"),
             info.get("yaw"), score, path, time.time(), json.dumps(cands), emb_blob, source, cell,
             json.dumps(marked_near or []), now))
    return True


def _prune_marked(c, session_id, only_student=None):
    """Remove marked students from pending cards' candidate lists. A card that
    had candidates and now has none is hidden (it was about a student who is
    already present)."""
    rows = c.execute("SELECT id, candidates, suggested_student_id, distance, image_path "
                     "FROM review_items WHERE session_id=? AND status='pending'",
                     (session_id,)).fetchall()
    for r in rows:
        try:
            cands = json.loads(r["candidates"]) if r["candidates"] else []
        except Exception:
            cands = []
        if not cands and r["suggested_student_id"]:         # v2 row: single suggestion
            cands = [{"student_id": r["suggested_student_id"], "distance": r["distance"]}]
        if not cands:
            continue
        keep = [x for x in cands
                if not (_marked(session_id, x["student_id"]) or
                        (only_student is not None and str(x["student_id"]) == str(only_student)))]
        if len(keep) == len(cands):
            continue
        if not keep:
            c.execute("UPDATE review_items SET status='auto_marked' WHERE id=?", (r["id"],))
            _remove_file(r["image_path"])
        else:
            c.execute("UPDATE review_items SET candidates=?, suggested_student_id=?, distance=? WHERE id=?",
                      (json.dumps(keep), keep[0]["student_id"], keep[0]["distance"], r["id"]))


def _merge_duplicates(c, session_id):
    """Collapse pending cards that show the same face (also cleans up duplicates
    created before this rule existed). Keeps the sharpest crop of each group."""
    rows = c.execute("SELECT id, score, embedding, seen, image_path, source, cell, created_at FROM review_items "
                     "WHERE session_id=? AND status='pending' AND embedding IS NOT NULL "
                     "ORDER BY score DESC", (session_id,)).fetchall()
    kept = []                                   # (row, views)
    for r in rows:
        raw = np.frombuffer(r["embedding"], dtype=np.float32)
        dim = 512 if len(raw) % 512 == 0 else len(raw)
        v = _views(r["embedding"], dim)
        if v is None:
            kept.append((r, None, 0))
            continue
        target = None
        for k, kv, kdim in kept:
            if kv is None or kdim != dim:
                continue
            dd = float(1.0 - np.max(kv @ v.T))
            same_spot = (r["source"] == "webcam" and k["source"] == "webcam" and r["cell"] == k["cell"]
                         and abs((r["created_at"] or 0) - (k["created_at"] or 0)) < 600
                         and dd < REVIEW_WEBCAM_MERGE_DIST)
            if dd < REVIEW_MERGE_DIST or same_spot:
                target = (k, kv)
                break
        if target is None:
            kept.append((r, v, dim))
            continue
        k, kv = target
        for i, (kr, kvv, kd) in enumerate(kept):          # fold this card's views into the kept one
            if kr["id"] == k["id"]:
                merged = np.vstack([kvv, v])[:REVIEW_VIEWS_PER_CARD]
                kept[i] = (kr, merged, kd)
                c.execute("UPDATE review_items SET seen=COALESCE(seen,1)+?, embedding=? WHERE id=?",
                          ((r["seen"] or 1), merged.astype(np.float32).tobytes(), kr["id"]))
        c.execute("DELETE FROM review_items WHERE id=?", (r["id"],))
        _remove_file(r["image_path"])


def drop_if_auto_marked(session_id, student_id):
    """Called right after a student is auto-marked present."""
    with _lock, _conn() as c:
        _prune_marked(c, session_id, only_student=student_id)


# --------------------------------------------------------------------------
# API for the teacher page
# --------------------------------------------------------------------------
@review_bp.get("/sessions")
def sessions():
    try:
        return jsonify(sessions=_sessions_fn() if _sessions_fn else [])
    except Exception as e:
        return jsonify(sessions=[], error=str(e))


@review_bp.get("/pending")
def pending():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify(error="session_id required"), 400

    with _lock, _conn() as c:
        _merge_duplicates(c, session_id)
        _prune_marked(c, session_id)
        rows = c.execute(
            "SELECT * FROM review_items WHERE session_id=? AND status='pending' "
            "ORDER BY (suggested_student_id IS NULL), distance ASC, created_at DESC",
            (session_id,)).fetchall()

        notices = c.execute("SELECT * FROM review_notices WHERE session_id=? ORDER BY last_seen DESC",
                            (session_id,)).fetchall()

    class_id = rows[0]["class_id"] if rows else (notices[0]["class_id"] if notices else None)
    students = []
    if class_id and _students_fn:
        try:
            students = _students_fn(class_id)
        except Exception:
            students = []
    names = {str(s["student_id"]): s.get("student_name") for s in students}

    items = []
    for r in rows:
        try:
            cands = json.loads(r["candidates"]) if r["candidates"] else []
        except Exception:
            cands = []
        if not cands and r["suggested_student_id"]:
            cands = [{"student_id": r["suggested_student_id"], "distance": r["distance"]}]
        for cd in cands:
            cd["student_name"] = names.get(str(cd["student_id"])) or str(cd["student_id"])
        top = cands[0] if cands else None
        try:
            mnear = json.loads(r["marked_near"]) if r["marked_near"] else []
        except Exception:
            mnear = []
        items.append({
            "already_marked_similar": mnear,
            "id": r["id"],
            "reason": r["reason"],
            "source": r["source"],
            "seen": r["seen"] or 1,
            "candidates": cands,
            "suggested_student_id": top["student_id"] if top else None,
            "suggested_student_name": top["student_name"] if top else None,
            "distance": top["distance"] if top else None,
            "blur": r["blur"], "width": r["width"], "yaw": r["yaw"],
            "created_at": r["created_at"],
            "image_url": f"/face/review/image/{r['id']}?t={int(r['created_at'] or 0)}",
        })
    alerts = [{
        "student_id": n["student_id"],
        "student_name": names.get(n["student_id"]) or n["student_name"] or n["student_id"],
        "source": n["source"], "count": n["count"], "last_seen": n["last_seen"],
    } for n in notices]
    return jsonify(count=len(items), class_id=class_id, students=students, items=items,
                   already_marked=alerts, roster=_build_roster(session_id))


def _build_roster(session_id):
    """Every enrolled student of the session's class with present/absent."""
    class_id, active = None, False
    try:
        for sess in (_sessions_fn() if _sessions_fn else []):
            if str(sess.get("session_id")) == str(session_id):
                class_id, active = sess.get("class_id"), True
                break
    except Exception as e:
        print(f"⚠️  roster: could not read active sessions: {e}")
    if class_id is None:                       # session already ended: fall back to review data
        with _conn() as c:
            r = c.execute("SELECT class_id FROM review_items WHERE session_id=? LIMIT 1", (session_id,)).fetchone()
            if not r:
                r = c.execute("SELECT class_id FROM review_notices WHERE session_id=? LIMIT 1", (session_id,)).fetchone()
        class_id = r["class_id"] if r else None

    students, error = [], None
    if class_id is not None and _students_fn:
        try:
            students = _students_fn(class_id)
        except Exception as e:
            error = f"could not read students for class {class_id}: {e}"
            print(f"⚠️  roster: {error}")
    out = []
    for x in students:
        sid = str(x["student_id"])
        row = {"student_id": sid, "student_name": x.get("student_name") or sid,
               "present": _marked(session_id, sid), "source": None, "at": None,
               "confirming": 0, "needed": 0}
        if _roster_info_fn:
            try:
                info = _roster_info_fn(session_id, sid) or {}
                row["locked_absent"] = bool(info.get("absent_locked")) and not row["present"]
                if row["present"]:
                    row["source"], row["at"] = info.get("source"), info.get("at")
                else:
                    row["confirming"], row["needed"] = info.get("confirming", 0), info.get("needed", 0)
            except Exception:
                pass
        out.append(row)
    present = sum(1 for x in out if x["present"])
    confirming = sum(1 for x in out if not x["present"] and x["confirming"] > 0)
    res = {"class_id": class_id, "active": active, "total": len(out), "present": present,
           "confirming": confirming, "absent": len(out) - present - confirming, "students": out}
    if _scan_fn:
        try:
            res["scans"] = _scan_fn(session_id) or {}
        except Exception:
            res["scans"] = {}
    if class_id is None:
        res["error"] = "session not found — is it still running?"
    elif error:
        res["error"] = error
    return res


@review_bp.post("/attendance")
def set_attendance():
    """Teacher flips a student: {"session_id","student_id","status":"present"|"absent","reviewer"}"""
    data = request.get_json(silent=True) or {}
    session_id, student_id = data.get("session_id"), data.get("student_id")
    status, reviewer = data.get("status"), data.get("reviewer")
    if not session_id or not student_id or status not in ("present", "absent"):
        return jsonify(error="session_id, student_id and status (present|absent) are required"), 400
    if not _set_status_fn:
        return jsonify(error="manual attendance is not enabled on this server"), 501

    info = _build_roster(session_id)
    if info.get("class_id") is None:
        return jsonify(error="session not found — is it still running?"), 404
    if str(student_id) not in {x["student_id"] for x in info["students"]}:
        return jsonify(error="student is not in this class"), 404

    try:
        warning = _set_status_fn(session_id, info["class_id"], str(student_id), status, reviewer)
    except Exception as e:
        return jsonify(error=f"Could not change attendance: {e}"), 502

    with _lock, _conn() as c:
        c.execute("INSERT INTO review_overrides (session_id, student_id, status, reviewer, at, warning) "
                  "VALUES (?,?,?,?,?,?)", (session_id, str(student_id), status, reviewer, time.time(), warning))
        if status == "present":
            _prune_marked(c, session_id, only_student=student_id)   # cards that only pointed at them go away
    return jsonify(ok=True, status=status, student_id=str(student_id), warning=warning)


@review_bp.get("/roster")
def roster():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify(error="session_id required"), 400
    return jsonify(_build_roster(session_id))


@review_bp.get("/image/<int:item_id>")
def image(item_id):
    with _conn() as c:
        row = c.execute("SELECT image_path FROM review_items WHERE id=?", (item_id,)).fetchone()
    if not row or not os.path.exists(row["image_path"]):
        abort(404)
    return send_file(row["image_path"], mimetype="image/jpeg")


@review_bp.post("/<int:item_id>/decide")
def decide(item_id):
    """Body: {"action": "confirm"|"reject"|"reassign", "student_id": "...", "reviewer": "..."}
       confirm  = top suggested student is right
       reassign = it is a different (or one of the other candidate) student
       reject   = dismiss (not a student / not them)"""
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    reviewer = data.get("reviewer")
    if action not in ("confirm", "reject", "reassign"):
        return jsonify(error="action must be confirm|reject|reassign"), 400

    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
        if not row:
            return jsonify(error="not found"), 404
        if row["status"] != "pending":
            return jsonify(error=f"already {row['status']}"), 409

        final_id, status = None, "rejected"
        if action == "confirm":
            if not row["suggested_student_id"]:
                return jsonify(error="no suggested student; use reassign and pick one"), 400
            final_id, status = row["suggested_student_id"], "confirmed"
        elif action == "reassign":
            final_id = data.get("student_id")
            if not final_id:
                return jsonify(error="student_id required"), 400
            status = "reassigned"

        if final_id and _mark_fn:
            try:
                _mark_fn(row["session_id"], row["class_id"], final_id)
            except Exception as e:      # item stays pending so it can be retried
                return jsonify(error=f"Could not mark attendance: {e}"), 502

        if final_id:  # keep crop for later re-enrollment
            d = os.path.join(REVIEW_DIR, "confirmed", _safe(final_id))
            os.makedirs(d, exist_ok=True)
            try:
                os.replace(row["image_path"], os.path.join(d, os.path.basename(row["image_path"])))
            except OSError:
                pass
        elif action == "reject":
            _remove_file(row["image_path"])   # embedding stays so this face isn't re-queued

        c.execute("UPDATE review_items SET status=?, final_student_id=?, reviewer=?, decided_at=? "
                  "WHERE id=?", (status, str(final_id) if final_id else None, reviewer, time.time(), item_id))

        # the student is present now: remove them from other cards' candidate lists
        if final_id:
            _prune_marked(c, row["session_id"], only_student=final_id)

    # teacher-verified face -> extra training sample for that student (best effort)
    if final_id and _learn_fn and row["embedding"]:
        try:
            _learn_fn(row["class_id"], str(final_id), _blob_to_emb(row["embedding"]))
        except Exception as e:
            print(f"⚠️  review learn error: {e}")
    return jsonify(ok=True, status=status, student_id=final_id)


def purge_old():
    """Privacy: delete old crops. Call at startup or on a timer."""
    cutoff = time.time() - REVIEW_RETENTION_DAYS * 86400
    with _lock, _conn() as c:
        for r in c.execute("SELECT id, image_path FROM review_items WHERE created_at<?", (cutoff,)).fetchall():
            _remove_file(r["image_path"])
        c.execute("DELETE FROM review_items WHERE created_at<?", (cutoff,))
        c.execute("DELETE FROM review_notices WHERE last_seen<?", (cutoff,))
