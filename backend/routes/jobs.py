import datetime
import uuid
from flask import Blueprint, jsonify, request
from backend.db import get_db_connection, dict_from_row

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.route("/api/jobs", methods=["GET"])
def get_jobs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT j.*, d.filename as dataset_filename, d.source_type
        FROM jobs j
        JOIN datasets d ON j.dataset_id = d.id
        ORDER BY j.created_at DESC
        """
    )
    jobs = [dict_from_row(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"jobs": jobs})


@jobs_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT j.*, d.filename as dataset_filename, d.source_type
        FROM jobs j
        JOIN datasets d ON j.dataset_id = d.id
        WHERE j.id = ?
        """,
        (job_id,)
    )
    job = dict_from_row(cursor.fetchone())
    conn.close()

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


@jobs_bp.route("/api/jobs/<job_id>/retry", methods=["POST"])
def retry_job(job_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    original_job = dict_from_row(cursor.fetchone())

    if not original_job:
        conn.close()
        return jsonify({"error": "Job not found"}), 404

    new_job_id = f"job_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT INTO jobs (id, dataset_id, status, stage, progress, frame_stride, max_frames, camera_config, created_at)
        VALUES (?, ?, 'queued', 'queued', 0.0, ?, ?, ?, ?)
        """,
        (
            new_job_id,
            original_job["dataset_id"],
            original_job.get("frame_stride", 2),
            original_job.get("max_frames"),
            original_job.get("camera_config", "configs/camera_default.yaml"),
            now_iso,
        )
    )
    conn.commit()
    conn.close()

    return jsonify({"job_id": new_job_id, "dataset_id": original_job["dataset_id"], "status": "queued"}), 201
