import os
import shutil
from typing import Tuple

DATA_DIR = "data"
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")


def ensure_directories() -> None:
    for path in [DATA_DIR, DATASETS_DIR, RESULTS_DIR, JOBS_DIR]:
        os.makedirs(path, exist_ok=True)


def get_dataset_dir(dataset_id: str) -> str:
    path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_result_dir(result_id: str) -> str:
    path = os.path.join(RESULTS_DIR, result_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_job_dir(job_id: str) -> str:
    path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_video(dataset_id: str, file_obj, filename: str) -> str:
    dataset_dir = get_dataset_dir(dataset_id)
    dest_path = os.path.join(dataset_dir, "source.mp4")
    file_obj.save(dest_path)

    # Write meta.json
    meta_path = os.path.join(dataset_dir, "meta.json")
    with open(meta_path, "w") as f:
        import json
        json.dump({"source_type": "upload", "original_filename": filename}, f)

    return dest_path
