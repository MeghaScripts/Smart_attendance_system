-- ============================================================================
-- Face Recognition Module — MySQL schema
-- Run once against your MySQL database before starting the service:
--   mysql -u root -p smart_attendance < schema.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS smart_attendance
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE smart_attendance;

-- ── Classes (referenced by students.class_id) ───────────────────────────────
-- If your main backend already owns a `classes` table, drop this and point
-- MYSQL_DATABASE at that database instead — this is only a minimal stand-in
-- so the FK on `students` has somewhere to point.
CREATE TABLE IF NOT EXISTS classes (
    class_id        INT AUTO_INCREMENT PRIMARY KEY,
    class_name      VARCHAR(100) NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ── Students known to the face recognition module ──────────────────────────
-- If your main Flask backend already owns a `students` table with a
-- compatible shape, point MYSQL_DATABASE at that same database and skip
-- creating this table — the module only needs: student_id (PK/unique),
-- student_name, class_id, email.
CREATE TABLE IF NOT EXISTS students (
    student_id      VARCHAR(64)  PRIMARY KEY,
    student_name    VARCHAR(255) NOT NULL,
    class_id        INT,
    department      VARCHAR(100) NOT NULL,
    semester        INT NOT NULL,
    section         VARCHAR(10)  NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    phone           VARCHAR(20),
    FOREIGN KEY (class_id) REFERENCES classes(class_id),
    INDEX idx_class (class_id)
) ENGINE=InnoDB;

-- ── Enrolled face embeddings (FaceNet512, 512-D) ────────────────────────────
-- `embeddings` stores every augmented-variant embedding as a JSON array of
-- 512-D float arrays. `embedding` keeps the first one for quick/legacy reads.
-- Display fields (name, email, class) are not duplicated here — they're
-- always read fresh from `students` by student_id.
CREATE TABLE IF NOT EXISTS face_embeddings (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    student_id         VARCHAR(64) NOT NULL UNIQUE,
    embeddings         JSON NOT NULL,
    embedding          JSON NOT NULL,
    model              VARCHAR(50) NOT NULL DEFAULT 'Facenet512',
    num_embeddings     INT NOT NULL DEFAULT 0,
    enrolled_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at         DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_face_embeddings_student
        FOREIGN KEY (student_id) REFERENCES students(student_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── Trained SVM classifiers, one per class ──────────────────────────────
-- The fitted (StandardScaler + SVC + LabelEncoder) pipeline is pickled and
-- stored on disk (see core/config.py: MODELS_DIR); this row just tracks
-- metadata + the file path so the API can report training status.
CREATE TABLE IF NOT EXISTS classifiers (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    class_id        INT NOT NULL UNIQUE,
    model_path      VARCHAR(500) NOT NULL,
    num_classes     INT NOT NULL,
    num_samples     INT NOT NULL,
    trained_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_class (class_id)
) ENGINE=InnoDB;

-- ── Optional local attendance mirror ────────────────────────────────────────
-- The module posts marks to BACKEND_URL/internal/attendance/mark (your main
-- Flask backend) as the source of truth. This table is only a local
-- audit log so the face module can answer "who did I mark, and when" even
-- if the backend call later fails, and is not required for recognition.
CREATE TABLE IF NOT EXISTS attendance_log (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL,
    class_id        INT NOT NULL,
    student_id      VARCHAR(64) NOT NULL,
    source          VARCHAR(20) NOT NULL,   -- webcam | cctv | photo
    confidence      FLOAT NOT NULL,
    svm_probability FLOAT NULL,
    backend_synced  TINYINT(1) NOT NULL DEFAULT 0,
    marked_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_class (class_id)
) ENGINE=InnoDB;
