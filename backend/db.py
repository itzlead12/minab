import os
import sqlite3
from typing import Dict, List, Optional, Any

DB_PATH = os.path.join("data", "minab.db")


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id           TEXT PRIMARY KEY,
            source_type  TEXT NOT NULL,
            filename     TEXT NOT NULL,
            stored_path  TEXT NOT NULL,
            has_telemetry INTEGER DEFAULT 0,
            mission_id   TEXT,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS missions (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            area_geojson TEXT,
            altitude_m   REAL,
            overlap_pct  REAL,
            speed_mps    REAL,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id           TEXT PRIMARY KEY,
            dataset_id   TEXT NOT NULL REFERENCES datasets(id),
            status       TEXT NOT NULL,
            stage        TEXT,
            progress     REAL DEFAULT 0.0,
            error        TEXT,
            result_id    TEXT,
            frame_stride INTEGER DEFAULT 2,
            max_frames   INTEGER,
            camera_config TEXT DEFAULT 'configs/camera_default.yaml',
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            finished_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS results (
            id                TEXT PRIMARY KEY,
            job_id            TEXT NOT NULL REFERENCES jobs(id),
            dataset_id        TEXT NOT NULL REFERENCES datasets(id),
            point_cloud_path  TEXT,
            mesh_ply_path     TEXT,
            mesh_obj_path     TEXT,
            trajectory_path   TEXT,
            preview_glb_path  TEXT,
            units             TEXT DEFAULT 'relative',
            created_at        TEXT NOT NULL
        );
    """
    )
    conn.commit()
    conn.close()


def dict_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None
