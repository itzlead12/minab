# Minab Web Dashboard (Level-1 Monocular 3D Reconstruction)

A minimal, auth-less local web dashboard for uploading monocular videos, running 3D reconstruction (`src/pipeline.py`), and viewing interactive 3D surface meshes, point clouds, and camera trajectories in the browser.

---

## 🚀 Quick Start

### 1. Run the Dashboard & Worker
From the project root directory:

```bash
python run_dev.py
```

Or via Flask directly:

```bash
python -m backend.app
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📂 Storage & Directory Layout

All database state and physical files reside under `data/`:

- `data/minab.db` — SQLite database storing datasets, jobs, results, and missions.
- `data/datasets/<dataset_id>/` — Uploaded videos (`source.mp4`), metadata (`meta.json`), and telemetry stub (`telemetry.csv`).
- `data/jobs/<job_id>/` — Job progress logs and state trackers (`progress.json`).
- `data/results/<result_id>/` — Reconstructed 3D artifacts:
  - `fused_point_cloud.ply` — Fused dense 3D point cloud
  - `reconstructed_mesh.ply` — Poisson surface mesh (PLY)
  - `reconstructed_mesh.obj` — Poisson surface mesh (OBJ)
  - `camera_trajectory.txt` — Camera pose trajectory (TUM format)

---

## 🛩️ UAV-Ready Architecture Seams (Level 2+ Ready)

This dashboard is architected so UAV integration drops in without a rewrite:

| Seam | Level-1 Today | Level 2+ (UAV Extension) |
|------|---------------|--------------------------|
| `datasets.source_type` | Always `"upload"` | `"uav"` for drone flight datasets |
| `datasets.telemetry.csv` | Stub file | Parsed MAVLink log / PX4 flight log |
| `datasets.mission_id` | `NULL` | Foreign key referencing `missions` table |
| `GET /api/telemetry/<dataset_id>` | STUB (returns `404`) | Returns parsed flight telemetry JSON |
| `POST /api/missions` | Database table ready | Creates UAV flight mission waypoint boundary |
| `results.units` | `"relative"` | `"meters"` (once metric scale is anchored) |
| Job worker `run_vision_pipeline` | Monocular video only | Passes `--telemetry` flag to pipeline |
| 3D Viewer | Arbitrary scale warning banner | Georeferenced metric scale axes |
