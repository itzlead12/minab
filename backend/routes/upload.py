import uuid
import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from backend.db import get_db_connection
from backend.services.storage import save_uploaded_video

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route("/api/upload", methods=["POST"])
def handle_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    job_id = f"job_{uuid.uuid4().hex[:10]}"

    # Save video to data/datasets/<dataset_id>/source.mp4
    stored_path = save_uploaded_video(dataset_id, file, filename)

    # Read options from form
    frame_stride = request.form.get("frame_stride", type=int, default=2)
    max_frames_raw = request.form.get("max_frames", type=str, default="").strip()
    max_frames = int(max_frames_raw) if max_frames_raw and max_frames_raw.isdigit() else None
    camera_config = request.form.get("camera_config", type=str, default="configs/camera_default.yaml")

    now_iso = datetime.datetime.utcnow().isoformat()

    conn = get_db_connection()
    # 1. Insert dataset
    conn.execute(
        """
        INSERT INTO datasets (id, source_type, filename, stored_path, has_telemetry, mission_id, created_at)
        VALUES (?, 'upload', ?, ?, 0, NULL, ?)
        """,
        (dataset_id, filename, stored_path, now_iso)
    )

    # 2. Insert job
    conn.execute(
        """
        INSERT INTO jobs (id, dataset_id, status, stage, progress, frame_stride, max_frames, camera_config, created_at)
        VALUES (?, ?, 'queued', 'queued', 0.0, ?, ?, ?, ?)
        """,
        (job_id, dataset_id, frame_stride, max_frames, camera_config, now_iso)
    )
    conn.commit()
    conn.close()

    return jsonify({"job_id": job_id, "dataset_id": dataset_id, "status": "queued"}), 201
