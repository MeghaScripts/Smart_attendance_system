"""
CCTV Service — Standalone IP camera integration.

Each class has its own dedicated IP camera (not NVR channels).
Camera IPs confirmed from NVR Function Panel → Manage Camera table.

Ports:
  - HTTP port 9008  → each camera's own web/management port (confirmed)
  - RTSP port 554   → standard RTSP (AEC-920TDNR2 /profile1 confirmed working)
  - NVR login 172.16.1.205:91 → NVR browser UI only, NOT camera port

IMPROVEMENTS:
  - AEC-920TDNR2 RTSP patterns added (/profile1, /profile2, /stream0, /stream1)
  - Reduced latency: buffer=1, timeout=2s, sleep=0.01s
  - Better error handling and frame capture
  - Capture resolution bumped from 640x480 -> config-driven 1920x1080
    (RTSP_CAPTURE_WIDTH/HEIGHT). At 640x480 faces beyond the front couple
    of rows in a ~60-student classroom fall below MIN_FACE_SIZE and can't
    be detected at all — resolution matters more than latency here.
"""
import os
import cv2
import threading
import time
import numpy as np
import urllib.request
import urllib.error
import urllib.parse
import base64
import http.cookiejar
from typing import Dict, Optional, Generator, Tuple

from core.config import face_settings

# ─── LATENCY OPTIMIZATION ────────────────────────────────────────────────────
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;2000000"  # REDUCED from 5000000 (5s→2s timeout)
)

# ─── Room → Camera mapping (from NVR Manage Camera table) ────────────────────
# IT person never needs to know RTSP URLs — just enter password in modal
# Add/update entries here when cameras change
ROOM_CAMERA_MAP = {
    "A320":  {"ip": "172.16.1.102", "http_port": 9008, "rtsp_port": 554},
    "A314":  {"ip": "172.16.1.105", "http_port": 9008, "rtsp_port": 554},
    "A317":  {"ip": "172.16.1.106", "http_port": 9008, "rtsp_port": 554},
    "A322":  {"ip": "172.16.1.107", "http_port": 9008, "rtsp_port": 554},
    "A324":  {"ip": "172.16.1.108", "http_port": 9008, "rtsp_port": 554},
    "A310":  {"ip": "172.16.1.109", "http_port": 9008, "rtsp_port": 554},
    "A312":  {"ip": "172.16.1.110", "http_port": 9008, "rtsp_port": 554},
    "A313":  {"ip": "172.16.1.111", "http_port": 9008, "rtsp_port": 554},
    "A304":  {"ip": "172.16.1.112", "http_port": 9008, "rtsp_port": 554},
    # Add remaining rooms after scrolling NVR camera list
    # "A315": {"ip": "172.16.1.1XX", "http_port": 9008, "rtsp_port": 554},
    # "A323": {"ip": "172.16.1.1XX", "http_port": 9008, "rtsp_port": 554},
    # "A318": {"ip": "172.16.1.1XX", "http_port": 9008, "rtsp_port": 554},
    # "A303": {"ip": "172.16.1.1XX", "http_port": 9008, "rtsp_port": 554},
    # CS-prefixed classes that share physical rooms
    "CS101": {
        "ip": "172.16.1.111", "http_port": 9008, "rtsp_port": 554,
        # Confirmed working directly against this camera — bypasses HTTP
        # snapshot guessing and RTSP pattern guessing entirely (see
        # start_stream: an explicit rtsp_url here is tried FIRST, fast).
        "rtsp_url": "rtsp://admin:123456@172.16.1.111:554/",
    },
}


def get_camera_for_room(class_id: str) -> dict:
    """Return camera config for a class. Empty dict if not mapped."""
    return ROOM_CAMERA_MAP.get(class_id, {})


DEFAULT_HTTP_PORTS = [9008, 80, 8080, 91]

_test_cancel_event = threading.Event()


def cancel_rtsp_test():
    _test_cancel_event.set()


# ─── HTTP snapshot URL templates ──────────────────────────────────────────────
SNAPSHOT_TEMPLATES = [
    "http://{ip}:{port}/cgi-bin/snapshot.cgi",
    "http://{ip}:{port}/snapshot.jpg",
    "http://{ip}:{port}/jpg/image.jpg",
    "http://{ip}:{port}/cgi-bin/snapshot.cgi?1",
    "http://{ip}:{port}/onvif/snapshot",
    "http://{ip}:{port}/capture",
    "http://{ip}:{port}/cgi-bin/snapshot.cgi?channel={ch}",
    "http://{ip}:{port}/cgi-bin/snapshot.cgi?chn={ch}",
    "http://{ip}:{port}/cgi-bin/snapshot.cgi?channel={ch}&subtype=0",
    "http://{ip}:{port}/snapshot/ch{ch:02d}.jpg",
    "http://{ip}:{port}/cgi-bin/currentpic.cgi?channel={ch}",
    "http://{ip}:{port}/ISAPI/Streaming/channels/{ch}01/picture",
]

TVT_SNAPSHOT_TEMPLATES = SNAPSHOT_TEMPLATES


def _decode_frame(data: bytes) -> Optional[np.ndarray]:
    if not data or len(data) < 100:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if (frame is not None and frame.size > 0) else None


def _try_url_basic(url, username, password, timeout=4):
    auth = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    req = urllib.request.Request(url, headers={
        "Authorization": auth, "User-Agent": "Mozilla/5.0",
        "Accept": "image/jpeg,image/*,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "image" in ct or (data and data[:2] == b'\xff\xd8'):
                return data
    except Exception:
        pass
    return None


def _try_url_digest(url, username, password, timeout=4):
    try:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, username, password)
        auth_handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
        opener = urllib.request.build_opener(auth_handler)
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "image/jpeg,image/*,*/*",
        })
        with opener.open(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "image" in ct or (data and data[:2] == b'\xff\xd8'):
                return data
    except Exception:
        pass
    return None


def _try_url_no_auth(url, timeout=4):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "image/jpeg,image/*,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "image" in ct or (data and data[:2] == b'\xff\xd8'):
                return data
    except Exception:
        pass
    return None


def _try_url_cookie(ip, port, username, password, channel=1, timeout=5):
    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        login_url = f"http://{ip}:{port}/RPC2_Login"
        login_body = (
            f'{{"method":"global.login","params":{{'
            f'"userName":"{username}","password":"{password}",'
            f'"clientType":"Web3.0"}},"id":1}}'
        ).encode()
        req = urllib.request.Request(login_url, data=login_body, headers={
            "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"
        })
        with opener.open(req, timeout=timeout) as resp:
            resp.read()
        snap_url = f"http://{ip}:{port}/cgi-bin/snapshot.cgi?channel={channel}"
        req2 = urllib.request.Request(snap_url, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "image/jpeg,image/*"
        })
        with opener.open(req2, timeout=timeout) as resp:
            data = resp.read()
            if data and data[:2] == b'\xff\xd8':
                return data
    except Exception:
        pass
    return None


def fetch_http_snapshot(ip, channel, username, password, port=9008):
    """Try all HTTP auth methods and URL patterns. Returns BGR frame or None."""
    ports_to_try = list(dict.fromkeys([port] + DEFAULT_HTTP_PORTS))
    for p in ports_to_try:
        for tmpl in SNAPSHOT_TEMPLATES:
            try:
                url = tmpl.format(ip=ip, port=p, ch=channel)
            except KeyError:
                continue
            for fetch_fn, kwargs in [
                (_try_url_no_auth, {}),
                (_try_url_basic,   {"username": username, "password": password}),
                (_try_url_digest,  {"username": username, "password": password}),
            ]:
                try:
                    data = fetch_fn(url, **kwargs)
                except TypeError:
                    continue
                frame = _decode_frame(data) if data else None
                if frame is not None:
                    print(f"  📸 HTTP snapshot OK [{fetch_fn.__name__}]: {url}")
                    return frame
    for p in ports_to_try:
        data = _try_url_cookie(ip, p, username, password, channel)
        frame = _decode_frame(data) if data else None
        if frame is not None:
            print(f"  📸 HTTP snapshot OK [cookie]: port {p}")
            return frame
    return None


def probe_http_snapshot(ip, channel, username, password, port=9008):
    """Probe all methods. Returns (success, description)."""
    ports_to_try = list(dict.fromkeys([port] + DEFAULT_HTTP_PORTS))
    for p in ports_to_try:
        for tmpl in SNAPSHOT_TEMPLATES:
            try:
                url = tmpl.format(ip=ip, port=p, ch=channel)
            except KeyError:
                continue
            for fetch_fn, kwargs in [
                (_try_url_no_auth, {}),
                (_try_url_basic,   {"username": username, "password": password}),
                (_try_url_digest,  {"username": username, "password": password}),
            ]:
                try:
                    data = fetch_fn(url, **kwargs)
                except TypeError:
                    continue
                if _decode_frame(data) is not None:
                    return True, f"{url} [{fetch_fn.__name__}]"
                elif data:
                    return False, f"{url} returned {len(data)}B but not valid JPEG"
    for p in ports_to_try:
        data = _try_url_cookie(ip, p, username, password, channel)
        if _decode_frame(data) is not None:
            return True, f"cookie auth on port {p}"
    return False, f"All HTTP patterns failed for {ip} (ports: {ports_to_try})"


# ─── RTSP helpers ─────────────────────────────────────────────────────────────

RTSP_PATTERNS = [
    # AEC-920TDNR2 specific (PRIMARY - discovered from camera config page)
    "rtsp://{user}:{pwd}@{ip}:{port}/profile1",
    "rtsp://{user}:{pwd}@{ip}:{port}/profile2",
    "rtsp://{user}:{pwd}@{ip}:{port}/stream0",
    "rtsp://{user}:{pwd}@{ip}:{port}/stream1",

    # Fallback patterns for other camera brands
    "rtsp://{user}:{pwd}@{ip}:{port}/ch{ch:02d}/0",
    "rtsp://{user}:{pwd}@{ip}:{port}/ch{ch:02d}/1",
    "rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{ch}01",
    "rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",
    "rtsp://{user}:{pwd}@{ip}:{port}/",
    "rtsp://{user}:{pwd}@{ip}:{port}/live",
]


def try_rtsp_connections(ip, port, channel, username, password):
    _test_cancel_event.clear()
    for pattern in RTSP_PATTERNS:
        if _test_cancel_event.is_set():
            return None, None
        try:
            url = pattern.format(user=username, pwd=password, ip=ip, port=port, ch=channel)
        except KeyError:
            continue
        print(f"  🔌 RTSP: {url.replace(password, '****')}")
        result_box = []

        def _attempt(u=url, rb=result_box):
            cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # LATENCY: Minimize buffer
            # RESOLUTION: config-driven, default 1920x1080. A 60-student
            # classroom needs full-res capture — 640x480 made faces beyond
            # the front rows undetectable (below MIN_FACE_SIZE).
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, face_settings.RTSP_CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, face_settings.RTSP_CAPTURE_HEIGHT)
            if cap.isOpened():
                ret, frm = cap.read()
                if ret and frm is not None:
                    rb.append((cap, u)); return
            cap.release()
            rb.append((None, None))

        t = threading.Thread(target=_attempt, daemon=True)
        t.start(); t.join(timeout=7)
        if result_box and result_box[0][0] is not None:
            cap, used = result_box[0]
            print(f"  ✅ RTSP connected: {used.replace(password, '****')}")
            return cap, used
    return None, None


def build_rtsp_url(ip, port, channel, username, password, brand="tvt"):
    templates = {
        "tvt":       f"rtsp://{username}:{password}@{ip}:{port}/profile1",  # UPDATED for AEC
        "hikvision": f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        "dahua":     f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "cpplus":    f"rtsp://{username}:{password}@{ip}:{port}/stream{channel}",
        "generic":   f"rtsp://{username}:{password}@{ip}:{port}/",
    }
    return templates.get(brand, templates["generic"])


# ─── Stream Manager ───────────────────────────────────────────────────────────

class CCTVStreamManager:
    def __init__(self):
        self._captures:      Dict[str, cv2.VideoCapture]     = {}
        self._latest_frames: Dict[str, Optional[np.ndarray]] = {}
        self._threads:       Dict[str, threading.Thread]     = {}
        self._running:       Dict[str, bool]                 = {}
        self._locks:         Dict[str, threading.Lock]       = {}
        self._rtsp_urls:     Dict[str, str]                  = {}
        self._snapshot_cfg:  Dict[str, dict]                 = {}

    def start_stream(self, class_id: str,
                     rtsp_url: str = None,
                     ip: str = None,
                     http_port: int = 9008,
                     rtsp_port: int = 554,
                     channel: int = 1,
                     username: str = "admin",
                     password: str = "",
                     brand: str = "tvt") -> tuple:

        if class_id in self._running and self._running[class_id]:
            return True, self._rtsp_urls.get(class_id, "already-running")

        self._locks[class_id]         = threading.Lock()
        self._running[class_id]       = True
        self._latest_frames[class_id] = None

        # ── Auto-lookup from room map (overrides manually entered IP) ─────
        cam = get_camera_for_room(class_id)
        if cam:
            print(f"  📍 Auto-mapped {class_id} → {cam['ip']}:{cam['http_port']}")
            ip        = cam["ip"]
            http_port = cam["http_port"]
            rtsp_port = cam["rtsp_port"]
            # A confirmed-working full RTSP URL in the map overrides
            # whatever was passed in (typed URL, guessed path, etc.) —
            # this is the fast path, tried before any guessing below.
            if cam.get("rtsp_url"):
                rtsp_url = cam["rtsp_url"]

        # ── 0. Known-good RTSP URL — try this FIRST, fast, before any
        #      HTTP-snapshot or RTSP-pattern guessing. This is the path
        #      that matters once a URL has been confirmed working (e.g.
        #      by testing in VLC / another dashboard) — no reason to burn
        #      minutes re-discovering it via the guess-every-pattern flow.
        if rtsp_url:
            print(f"⚡ Trying known RTSP URL directly for {class_id}…")
            result_box = []

            def _try_known(u=rtsp_url, rb=result_box):
                c = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FRAME_WIDTH, face_settings.RTSP_CAPTURE_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, face_settings.RTSP_CAPTURE_HEIGHT)
                if c.isOpened():
                    ret, frm = c.read()
                    if ret and frm is not None:
                        rb.append((c, u)); return
                c.release(); rb.append((None, None))

            t = threading.Thread(target=_try_known, daemon=True)
            t.start(); t.join(timeout=6)  # short timeout — this URL is supposed to just work
            if result_box and result_box[0][0] is not None:
                cap, used_url = result_box[0]
                print(f"✅ Known RTSP URL connected instantly for {class_id}")
                self._captures[class_id]  = cap
                self._rtsp_urls[class_id] = used_url

                def capture_loop(cid=class_id):
                    failures = 0
                    while self._running.get(cid, False):
                        ret, frm = cap.read()
                        if ret and frm is not None:
                            with self._locks[cid]:
                                self._latest_frames[cid] = frm.copy()
                            failures = 0
                        else:
                            failures += 1
                            if failures > 50:
                                self._running[cid] = False
                                break
                            time.sleep(0.01)
                    cap.release()

                th = threading.Thread(target=capture_loop, daemon=True)
                self._threads[class_id] = th
                th.start()
                return True, used_url
            print(f"⚠️  Known RTSP URL failed for {class_id} — falling back to guessing")

        # ── 1. HTTP snapshot (port 9008 first) ────────────────────────────
        if ip and password:
            print(f"🌐 Trying HTTP snapshot for {class_id} ({ip}:{http_port})…")
            frame = fetch_http_snapshot(ip, channel, username, password, port=http_port)
            if frame is not None:
                self._snapshot_cfg[class_id] = {
                    "ip": ip, "channel": channel,
                    "username": username, "password": password,
                    "port": http_port,
                }
                with self._locks[class_id]:
                    self._latest_frames[class_id] = frame.copy()
                self._rtsp_urls[class_id] = "http-snapshot"
                print(f"✅ HTTP snapshot mode active for {class_id}")

                def http_loop(cid=class_id):
                    cfg = self._snapshot_cfg.get(cid, {})
                    while self._running.get(cid, False):
                        if cfg:
                            f = fetch_http_snapshot(
                                cfg["ip"], cfg["channel"],
                                cfg["username"], cfg["password"],
                                port=cfg.get("port", 9008),
                            )
                            if f is not None:
                                with self._locks[cid]:
                                    self._latest_frames[cid] = f.copy()
                        time.sleep(1.5)

                t = threading.Thread(target=http_loop, daemon=True)
                self._threads[class_id] = t
                t.start()
                return True, "http-snapshot"

        # ── 2. RTSP fallback ──────────────────────────────────────────────
        cap = None
        used_url = rtsp_url

        if rtsp_url:
            result_box = []
            def _try_direct(u=rtsp_url, rb=result_box):
                c = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FRAME_WIDTH, face_settings.RTSP_CAPTURE_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, face_settings.RTSP_CAPTURE_HEIGHT)
                if c.isOpened():
                    ret, frm = c.read()
                    if ret and frm is not None:
                        rb.append((c, u)); return
                c.release(); rb.append((None, None))
            t = threading.Thread(target=_try_direct, daemon=True)
            t.start(); t.join(timeout=8)
            if result_box and result_box[0][0]:
                cap, used_url = result_box[0]

        if cap is None and ip and password:
            cap, used_url = try_rtsp_connections(ip, rtsp_port, channel, username, password)

        if cap is not None:
            self._captures[class_id]  = cap
            self._rtsp_urls[class_id] = used_url

            def capture_loop(cid=class_id):
                failures = 0
                while self._running.get(cid, False):
                    ret, frm = cap.read()
                    if ret and frm is not None:
                        with self._locks[cid]:
                            self._latest_frames[cid] = frm.copy()
                        failures = 0
                    else:
                        failures += 1
                        if failures > 50:
                            self._running[cid] = False
                            break
                        time.sleep(0.01)  # LATENCY: Reduced from 0.05 for faster capture
                cap.release()

            t = threading.Thread(target=capture_loop, daemon=True)
            self._threads[class_id] = t
            t.start()
            return True, used_url

        # ── 3. Blind mode — retry on every process-now call ───────────────
        if ip and password:
            self._snapshot_cfg[class_id] = {
                "ip": ip, "channel": channel,
                "username": username, "password": password,
                "port": http_port,
            }
            self._rtsp_urls[class_id] = "pending"
            print(f"⚠️  No connection yet for {class_id} — will retry on process-now")
            return True, "pending"

        self._running[class_id] = False
        return False, ""

    def stop_stream(self, class_id: str):
        self._running[class_id] = False
        self._snapshot_cfg.pop(class_id, None)

    def get_latest_frame(self, class_id: str) -> Optional[np.ndarray]:
        lock = self._locks.get(class_id)
        if not lock:
            return None
        with lock:
            f = self._latest_frames.get(class_id)
            return f.copy() if f is not None else None

    def get_snapshot_cfg(self, class_id: str) -> Optional[dict]:
        return self._snapshot_cfg.get(class_id)

    def is_running(self, class_id: str) -> bool:
        return self._running.get(class_id, False)

    def mjpeg_generator(self, class_id: str) -> Generator:
        while self.is_running(class_id):
            frame = self.get_latest_frame(class_id)
            if frame is None:
                time.sleep(0.1)
                continue
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            time.sleep(0.05)


cctv_manager = CCTVStreamManager()