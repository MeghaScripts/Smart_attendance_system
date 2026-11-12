# Face Recognition Module (Flask + MySQL)


Standalone service handling face-recognition attendance for a class via
**webcam**, **CCTV (RTSP/HTTP snapshot)**, and **group photo upload**.
Tuned by default for a single class of up to **60 enrolled students**.

## What changed for 60-student scale (vs. the original build)

| Area | Before | Now | Why |
|---|---|---|---|
| CCTV capture resolution | hardcoded 640x480 | `RTSP_CAPTURE_WIDTH/HEIGHT` = 1920x1080 | At 640x480, faces beyond the front rows in a 60-student room fell below `MIN_FACE_SIZE` and were undetectable. |
| Photo-attendance upscale | flat 1280px floor | scales with `EXPECTED_CLASS_SIZE` (default 60) / actual enrolled count | A flat 1280px floor gives ~15-25px/face in a 60-person group photo — below `MIN_FACE_SIZE`. Target width now scales toward ~3200px for 60 faces. |
| `MATCH_MARGIN` (cosine ambiguity gap) | 0.06 | 0.04 | With more enrolled students, the raw-cosine runner-up sits closer to the winner more often. The SVM cross-check (not this margin) is what actually resolves ambiguity at this class size, so the margin no longer needs to be as strict. |
| `EXPECTED_CLASS_SIZE` config knob | n/a | new (default 60) | Lets the photo pipeline size its upscale target instead of guessing. |

Everything else (SVM-on-FaceNet512 architecture, 12-variant enrollment
augmentation, liveness checks, session-based multi-frame confirmation) was
already built to scale to 60 students without changes — a linear-kernel SVM
trained on 60 classes x 12 samples = 720 samples trains in well under a
second.

## Project layout

```
face_recognition_module/
├── main.py                       # Flask app + background CCTV recognition loop
├── db.py                         # MySQL connection pool + query helpers
├── schema.sql                    # classes / students / face_embeddings / classifiers / attendance_log
├── stub_backend.py                # Fake main-backend for local testing (catches attendance marks)
├── requirements.txt
├── Dockerfile
├── .env.example
├── core/
│   └── config.py                 # All tunables, env-driven
├── services/
│   ├── embedding_service.py       # Augmentation, FaceNet512 embedding, cosine matcher
│   ├── classifier_service.py      # Per-class SVM training/prediction
│   ├── cctv_service.py            # RTSP/HTTP snapshot camera integration
│   ├── session_manager.py         # Per-class recognition session state
│   ├── liveness_service.py        # Anti-spoofing (texture + eye checks)
│   └── photo_attendance_service.py# Group-photo attendance
└── api/
    └── face_routes.py             # All /face/* HTTP endpoints
```

## Setup

1. **MySQL**
   ```bash
   mysql -u root -p < schema.sql
   ```
   This creates a minimal `classes`/`students` pair. If your real backend
   already owns these tables, point `MYSQL_DATABASE` at that database and
   skip creating them here — the module only reads
   `student_id, student_name, class_id, email` from `students`.

2. **Environment**
   ```bash
   cp .env.example .env
   # edit MYSQL_PASSWORD, BACKEND_URL, CLASS_CAMERA_MAP, etc.
   ```

3. **Install**
   ```bash
   python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

4. **Run the stub backend** (only needed if your real backend isn't running yet)
   ```bash
   python stub_backend.py
   # listens on http://localhost:5000, logs every attendance mark it receives
   # GET http://localhost:5000/internal/attendance/all to see everything so far
   ```

5. **Run the face module**
   ```bash
   python main.py
   # listens on http://localhost:8001
   ```

   Or with Docker:
   ```bash
   docker build -t face-module .
   docker run -p 8001:8001 --env-file .env face-module
   ```

## Testing with 60 students end-to-end

1. **Seed 60 students** into `classes`/`students` (via your real backend, or
   directly with SQL) so each has a `student_id`, `student_name`, `email`,
   and the same `class_id`.

2. **Enroll each student's photo**:
   ```bash
   curl -X POST http://localhost:8001/face/enroll \
     -F "student_id=STU001" \
     -F "photo=@student1.jpg"
   ```
   Repeat for all 60 — each call retrains the class's SVM (fast, <1s even
   at 60 classes). Watch the console: it'll print
   `✅ SVM trained for <class_id>: 60 students, 720 training samples` once
   everyone's in.

3. **Start a recognition session**:
   ```bash
   curl -X POST http://localhost:8001/face/session/start \
     -H "Content-Type: application/json" \
     -d '{"session_id":"sess001","class_id":"CS101","teacher_id":"T1","use_cctv":true,"nvr_ip":"172.16.1.111","nvr_password":"your_camera_password"}'
   ```

4. **Webcam frame test** (PowerShell example — note the field is `class_id`,
   not `classroom_id`, to match the current schema):
   ```powershell
   $imagePath = "C:\Users\megha\OneDrive\Pictures\student1.jpg"
   $bytes = [System.IO.File]::ReadAllBytes($imagePath)
   $base64 = [System.Convert]::ToBase64String($bytes)
   $dataUrl = "data:image/jpeg;base64,$base64"

   $body = @{
       class_id   = "CS101"
       session_id = "sess001"
       frame_b64  = $dataUrl
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://localhost:8001/face/process-frame" -Method Post -Body $body -ContentType "application/json"
   ```

5. **Group photo test** (all 60 students in one shot):
   ```bash
   curl -X POST http://localhost:8001/face/photo-attendance \
     -F "session_id=sess001" \
     -F "class_id=CS101" \
     -F "photo=@classroom_group_photo.jpg"
   ```
   Use the phone's native resolution, not a compressed copy — the service
   upscales further internally but can't recover detail lost to compression.

6. **Verify marks landed**:
   ```bash
   curl http://localhost:5000/internal/attendance/all
   ```

## Web console (frontend)

A single-file web UI lives in `frontend/index.html` and is served by the module itself:

```
python main.py
# open http://localhost:8001/ui
```

It covers live webcam/CCTV sessions, group-photo attendance, student enrollment (4-10 photos each, single or bulk by folder), classifier status/retrain, and an attendance-records table with CSV export. Because it is served from the same origin, no CORS setup is needed and the browser allows webcam access on `localhost`.

If you host the page somewhere else, add that origin to `ALLOWED_ORIGINS` in `.env` and set the module URL on the Settings tab.

The Records tab reads a new read-only endpoint, `GET /face/attendance-log?class_id=&session_id=&limit=`, which returns rows from the local `attendance_log` table.

## Notes / things worth knowing before a demo

- `CLASS_CAMERA_MAP` in `.env` and `ROOM_CAMERA_MAP` in
  `services/cctv_service.py` are two separate mechanisms — the hardcoded
  dict in `cctv_service.py` is what's actually used for camera auto-lookup
  today. Add new rooms there.
- If CPU-only inference is too slow with 60 faces in a single CCTV frame,
  either run TensorFlow with GPU support, or switch the camera to its
  `/profile2` sub-stream instead of dropping `RTSP_CAPTURE_WIDTH/HEIGHT`
  back down — resolution loss costs you more accuracy than it saves latency.
- `schema.sql`'s `students.class_id` is typed `INT` (FK to `classes.class_id`).
  If your real backend uses a string class identifier like `"CS101"`
  directly (as the CCTV room map does), make sure both sides agree on the
  same type/value before wiring this up to production data.
