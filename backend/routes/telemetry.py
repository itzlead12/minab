from flask import Blueprint, jsonify
from backend.db import get_db_connection, dict_from_row

telemetry_bp = Blueprint("telemetry", __name__)


@telemetry_bp.route("/api/telemetry/<dataset_id>", methods=["GET"])
def get_telemetry(dataset_id: str):
    """
    UAV-Ready Stub Endpoint.
    For Level 1 uploads, telemetry does not exist, so return a clean 404.
    Level 2+ will replace this handler to return parsed MAVLink log JSON.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    ds = dict_from_row(cursor.fetchone())
    conn.close()

    if not ds:
        return jsonify({"error": "Dataset not found"}), 404

    # Return 404 as specified for non-telemetry upload datasets
    return jsonify({
        "error": "No telemetry data recorded for upload datasets",
        "dataset_id": dataset_id,
        "has_telemetry": ds.get("has_telemetry", 0),
        "source_type": ds.get("source_type", "upload")
    }), 404
