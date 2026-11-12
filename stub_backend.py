"""
Stub backend — catches POST /internal/attendance/mark so you can test the
face recognition module end-to-end without your real Flask/main backend
running. Point BACKEND_URL at this (http://localhost:5000) in .env while
testing 60-student enrollment + CCTV/webcam/photo recognition locally.
"""
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

received_marks = []


@app.route("/internal/attendance/mark", methods=["POST"])
def mark_attendance():
    body = request.get_json(force=True)
    body["received_at"] = datetime.utcnow().isoformat()
    received_marks.append(body)

    print(f"[STUB BACKEND] Attendance marked: "
          f"student_id={body.get('student_id')} "
          f"status={body.get('status')} "
          f"source={body.get('source')} "
          f"confidence={body.get('confidence')} "
          f"session_id={body.get('session_id')}")

    return jsonify({"success": True, "message": "Attendance recorded (stub)"}), 200


@app.route("/internal/attendance/all", methods=["GET"])
def list_marks():
    return jsonify(received_marks)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "stub_backend"})


if __name__ == "__main__":
    print("Stub backend running on http://localhost:5000")
    print("Catches POST /internal/attendance/mark and logs it.")
    print("GET /internal/attendance/all to see everything received so far.")
    app.run(host="0.0.0.0", port=5000, debug=False)
