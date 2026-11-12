"""
Recognition Session Manager — handles concurrent sessions across classes.
Each class_id runs an independent recognition loop with its own
embeddings cache. Thread-safe (Flask + background CCTV thread both touch it).
"""
import threading
import time
from datetime import datetime
from typing import Dict, Set

from core.config import face_settings


class RecognitionSessionManager:
    def __init__(self):
        self._lock = threading.RLock()
        # class_id -> session metadata
        self.active_sessions: Dict[str, dict] = {}
        # session_id -> set of already-recognized student_ids
        self.recognized_this_session: Dict[str, Set[str]] = {}
        # class_id -> cached embeddings (refreshed on session start)
        self.embeddings_cache: Dict[str, list] = {}
        # session_id -> {student_id: consecutive-match count}
        self.pending_confirmations: Dict[str, Dict[str, int]] = {}
        # session_id -> {student_id: {"source": str, "at": epoch}}  (how/when marked present)
        self.mark_info: Dict[str, Dict[str, dict]] = {}
        # session_id -> students the TEACHER set to absent: recognition must not flip them back
        self.absent_locked: Dict[str, Set[str]] = {}
        # session_id -> {source: {faces, matched, sent_to_review, already_marked, at}}  (last scan per source)
        self.scans: Dict[str, Dict[str, dict]] = {}
        # class_id -> lock so the background loop and a manual "Scan now" never process at the same time
        self._scan_locks: Dict[str, threading.Lock] = {}

    def start_session(self, class_id: str, session_id: str, teacher_id: str):
        with self._lock:
            self.active_sessions[class_id] = {
                "session_id": session_id,
                "teacher_id": teacher_id,
                "class_id": class_id,
                "started_at": datetime.utcnow(),
                "frames_processed": 0,
                "faces_recognized": 0,
            }
            self.recognized_this_session[session_id] = set()
            self.pending_confirmations[session_id] = {}
            self.mark_info[session_id] = {}
            self.absent_locked[session_id] = set()
            self.scans[session_id] = {}
        print(f"✅ Recognition session started: class={class_id}, session={session_id}")

    def end_session(self, class_id: str):
        with self._lock:
            session = self.active_sessions.pop(class_id, None)
            if session:
                session_id = session["session_id"]
                self.recognized_this_session.pop(session_id, None)
                self.pending_confirmations.pop(session_id, None)
                self.mark_info.pop(session_id, None)
                self.absent_locked.pop(session_id, None)
                self.scans.pop(session_id, None)
                self.embeddings_cache.pop(class_id, None)
        if session:
            print(f"🛑 Recognition session ended: class={class_id}")

    def register_candidate_match(self, session_id: str, student_id: str) -> bool:
        """
        Call once per frame for each student matched above threshold.
        Returns True only once the same student has matched in
        CONFIRMATION_FRAMES separate frame-processing calls for this
        session — the caller should only mark attendance when this
        returns True.
        """
        with self._lock:
            counts = self.pending_confirmations.setdefault(session_id, {})
            counts[student_id] = counts.get(student_id, 0) + 1
            return counts[student_id] >= face_settings.CONFIRMATION_FRAMES

    def is_active(self, class_id: str) -> bool:
        return class_id in self.active_sessions

    def get_session(self, class_id: str) -> dict:
        return self.active_sessions.get(class_id)

    def get_session_by_id(self, session_id: str) -> dict:
        for s in list(self.active_sessions.values()):
            if s["session_id"] == session_id:
                return s
        return None

    def already_recognized(self, session_id: str, student_id: str) -> bool:
        return student_id in self.recognized_this_session.get(session_id, set())

    def mark_recognized(self, session_id: str, student_id: str, source: str = None):
        with self._lock:
            if session_id in self.recognized_this_session:
                self.recognized_this_session[session_id].add(student_id)
                self.mark_info.setdefault(session_id, {})[student_id] = {
                    "source": source, "at": time.time(),
                }

    def scan_lock(self, class_id: str) -> threading.Lock:
        with self._lock:
            return self._scan_locks.setdefault(str(class_id), threading.Lock())

    # ── manual teacher overrides ─────────────────────────────────────────
    def unmark_recognized(self, session_id: str, student_id: str):
        """Teacher changed present -> absent."""
        with self._lock:
            self.recognized_this_session.get(session_id, set()).discard(student_id)
            self.mark_info.get(session_id, {}).pop(student_id, None)
            self.pending_confirmations.get(session_id, {}).pop(student_id, None)

    def lock_absent(self, session_id: str, student_id: str):
        with self._lock:
            self.absent_locked.setdefault(session_id, set()).add(student_id)

    def clear_absent_lock(self, session_id: str, student_id: str):
        with self._lock:
            self.absent_locked.get(session_id, set()).discard(student_id)

    def is_absent_locked(self, session_id: str, student_id: str) -> bool:
        return student_id in self.absent_locked.get(session_id, set())

    def record_scan(self, session_id: str, source: str, stats: dict):
        """Remember how many faces the last frame/photo contained and what happened to them."""
        if not session_id or not stats:
            return
        with self._lock:
            self.scans.setdefault(session_id, {})[source] = {**stats, "at": time.time()}

    def scan_info(self, session_id: str) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self.scans.get(session_id, {}).items()}

    def roster_info(self, session_id: str, student_id: str) -> dict:
        """For the review dashboard roster: how the student was marked, or how
        far along the multi-frame confirmation is."""
        with self._lock:
            info = dict(self.mark_info.get(session_id, {}).get(student_id) or {})
            info["absent_locked"] = student_id in self.absent_locked.get(session_id, set())
            info["confirming"] = self.pending_confirmations.get(session_id, {}).get(student_id, 0)
            info["needed"] = face_settings.CONFIRMATION_FRAMES
            return info

    def cache_embeddings(self, class_id: str, embeddings: list):
        with self._lock:
            self.embeddings_cache[class_id] = embeddings

    def get_cached_embeddings(self, class_id: str) -> list:
        return self.embeddings_cache.get(class_id, [])

    def increment_frame_count(self, class_id: str):
        with self._lock:
            if class_id in self.active_sessions:
                self.active_sessions[class_id]["frames_processed"] += 1

    def increment_recognition_count(self, class_id: str):
        with self._lock:
            if class_id in self.active_sessions:
                self.active_sessions[class_id]["faces_recognized"] += 1

    def get_all_active(self) -> list:
        return list(self.active_sessions.values())

    def clear_all_embedding_caches(self):
        """Wipe every class embedding cache so the next frame re-fetches
        fresh embeddings from MySQL. Call this after any re-enroll."""
        with self._lock:
            self.embeddings_cache.clear()
        print("🔄 All embedding caches cleared (re-enroll triggered)")

    def reload_embeddings_for_active_sessions(self):
        """Re-fetch embeddings from MySQL for every currently active
        class. Called after clear_all_embedding_caches so webcam
        sessions don't stall waiting for the next /session/start."""
        from services.embedding_service import FaceEmbeddingService
        service = FaceEmbeddingService()
        for class_id in list(self.active_sessions.keys()):
            try:
                fresh = service.get_class_embeddings(class_id)
                with self._lock:
                    self.embeddings_cache[class_id] = fresh
                print(f"🔄 Reloaded {len(fresh)} embeddings for {class_id}")
            except Exception as e:
                print(f"⚠️  Could not reload embeddings for {class_id}: {e}")


# Singleton shared across the face module
session_manager = RecognitionSessionManager()
