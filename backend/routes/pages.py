from flask import Blueprint, render_template, abort
from backend.db import get_db_connection, dict_from_row

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
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
    jobs = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    return render_template("index.html", jobs=jobs)


@pages_bp.route("/jobs/<job_id>")
def job_page(job_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT j.*, d.filename as dataset_filename, d.source_type, d.has_telemetry
        FROM jobs j
        JOIN datasets d ON j.dataset_id = d.id
        WHERE j.id = ?
        """,
        (job_id,)
    )
    job = dict_from_row(cursor.fetchone())

    if not job:
        conn.close()
        abort(404, description="Job not found")

    result = None
    if job.get("result_id"):
        cursor.execute("SELECT * FROM results WHERE id = ?", (job["result_id"],))
        result = dict_from_row(cursor.fetchone())

    conn.close()
    return render_template("job.html", job=job, result=result)
