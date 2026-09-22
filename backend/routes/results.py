import os
import re
from flask import Blueprint, jsonify, send_from_directory, abort
from backend.db import get_db_connection, dict_from_row
from backend.services.storage import RESULTS_DIR

results_bp = Blueprint("results", __name__)

ALLOWED_RESULT_FILES = {
    "fused_point_cloud.ply",
    "reconstructed_mesh.ply",
    "reconstructed_mesh.obj",
    "preview.glb",
    "camera_trajectory.txt",
    "camera_trajectory.json",
    "input_assessment.json",
}


@results_bp.route("/api/results/<result_id>", methods=["GET"])
def get_result(result_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results WHERE id = ?", (result_id,))
    res = dict_from_row(cursor.fetchone())
    conn.close()

    if not res:
        return jsonify({"error": "Result not found"}), 404

    return jsonify(res)


@results_bp.route("/files/<result_id>/<filename>", methods=["GET"])
def serve_result_file(result_id: str, filename: str):
    # Validate result_id
    if not re.match(r"^[a-zA-Z0-9_-]+$", result_id):
        abort(400, description="Invalid result ID")

    # Validate filename allowlist
    if filename not in ALLOWED_RESULT_FILES:
        abort(403, description="File access not permitted")

    result_dir = os.path.abspath(os.path.join(RESULTS_DIR, result_id))

    # Security check: ensure target directory exists and is within RESULTS_DIR
    abs_results_dir = os.path.abspath(RESULTS_DIR)
    if not result_dir.startswith(abs_results_dir):
        abort(403, description="Path traversal denied")

    file_path = os.path.join(result_dir, filename)
    if not os.path.exists(file_path):
        abort(404, description=f"File {filename} not found for result {result_id}")

    return send_from_directory(result_dir, filename)
