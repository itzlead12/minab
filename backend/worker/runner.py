import os
import json
import time
import uuid
import datetime
import threading
import logging
from backend.db import get_db_connection
from backend.services.storage import get_result_dir, get_job_dir
from backend.worker.vision_adapter import run_vision_pipeline

logger = logging.getLogger("minab.worker.runner")


def claim_next_job():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
    )
    job = cursor.fetchone()
    if not job:
        conn.close()
        return None

    job_id = job["id"]
    now_iso = datetime.datetime.utcnow().isoformat()

    cursor.execute(
        "UPDATE jobs SET status = 'running', stage = 'extracting_frames', progress = 0.05, started_at = ? WHERE id = ? AND status = 'queued'",
        (now_iso, job_id)
    )
    conn.commit()

    if cursor.rowcount == 0:
        conn.close()
        return None

    # Fetch updated job
    cursor.execute("SELECT j.*, d.stored_path as dataset_path FROM jobs j JOIN datasets d ON j.dataset_id = d.id WHERE j.id = ?", (job_id,))
    updated_job = cursor.fetchone()
    conn.close()
    return updated_job


def process_job(job):
    job_id = job["id"]
    dataset_id = job["dataset_id"]
    video_path = job["dataset_path"]
    frame_stride = job["frame_stride"] or 2
    max_frames = job["max_frames"]
    camera_config = job["camera_config"] or "configs/camera_default.yaml"

    result_id = f"res_{uuid.uuid4().hex[:10]}"
    result_dir = get_result_dir(result_id)
    job_dir = get_job_dir(job_id)
    progress_json_path = os.path.join(job_dir, "progress.json")

    logger.info(f"Starting execution for job {job_id} (dataset: {dataset_id})")

    # Start progress watcher thread
    stop_event = threading.Event()

    def watch_progress():
        last_mtime = 0
        while not stop_event.is_set():
            if os.path.exists(progress_json_path):
                try:
                    mtime = os.path.getmtime(progress_json_path)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        with open(progress_json_path, "r") as f:
                            data = json.load(f)
                        stage = data.get("stage", "running")
                        progress = data.get("progress", 0.1)

                        conn = get_db_connection()
                        conn.execute(
                            "UPDATE jobs SET stage = ?, progress = ? WHERE id = ?",
                            (stage, progress, job_id)
                        )
                        conn.commit()
                        conn.close()
                except Exception as ex:
                    logger.debug(f"Progress parse warning: {ex}")
            time.sleep(0.5)

    watcher = threading.Thread(target=watch_progress, daemon=True)
    watcher.start()

    try:
        run_vision_pipeline(
            video_path=video_path,
            output_dir=result_dir,
            camera_config=camera_config,
            frame_stride=frame_stride,
            max_frames=max_frames,
            progress_json_path=progress_json_path,
        )

        stop_event.set()
        watcher.join(timeout=1.0)

        # Build results DB entry
        ply_path = os.path.join(result_dir, "fused_point_cloud.ply")
        mesh_ply_path = os.path.join(result_dir, "reconstructed_mesh.ply")
        mesh_obj_path = os.path.join(result_dir, "reconstructed_mesh.obj")
        traj_path = os.path.join(result_dir, "camera_trajectory.txt")
        preview_glb_path = os.path.join(result_dir, "preview.glb")

        now_iso = datetime.datetime.utcnow().isoformat()
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO results (
                id, job_id, dataset_id, point_cloud_path, mesh_ply_path,
                mesh_obj_path, trajectory_path, preview_glb_path, units, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, job_id, dataset_id,
                ply_path if os.path.exists(ply_path) else None,
                mesh_ply_path if os.path.exists(mesh_ply_path) else None,
                mesh_obj_path if os.path.exists(mesh_obj_path) else None,
                traj_path if os.path.exists(traj_path) else None,
                preview_glb_path if os.path.exists(preview_glb_path) else None,
                "relative",
                now_iso
            )
        )

        conn.execute(
            "UPDATE jobs SET status = 'done', stage = 'done', progress = 1.0, result_id = ?, finished_at = ? WHERE id = ?",
            (result_id, now_iso, job_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Job {job_id} successfully completed. Result ID: {result_id}")

    except Exception as e:
        stop_event.set()
        now_iso = datetime.datetime.utcnow().isoformat()
        err_msg = str(e)
        logger.error(f"Job {job_id} failed: {err_msg}")
        conn = get_db_connection()
        conn.execute(
            "UPDATE jobs SET status = 'failed', stage = 'failed', error = ?, finished_at = ? WHERE id = ?",
            (err_msg, now_iso, job_id)
        )
        conn.commit()
        conn.close()


def worker_loop(poll_interval=1.0, stop_event=None):
    logger.info("Worker loop started.")
    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            job = claim_next_job()
            if job:
                process_job(job)
            else:
                time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            time.sleep(poll_interval)


def start_worker_thread():
    thread = threading.Thread(target=worker_loop, daemon=True, name="MinabBackgroundWorker")
    thread.start()
    return thread
