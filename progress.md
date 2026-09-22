# ምናብ | Minab Project — Progress & Architecture Summary

**Project Name**: ምናብ (Minab) — Monocular 3D Spatial Reconstruction Platform  
**Current Milestone**: Level-1 Auth-less Web Dashboard Proof-of-Concept (POC)  
**Last Updated**: September 19, 2026  

---

## 🎯 Executive Summary

The **Minab** project is an end-to-end 3D spatial reconstruction system designed to turn monocular video (phone walkarounds, drone feeds) into dense 3D point clouds, camera trajectories, and textured surface meshes.

We have successfully completed **Level-1**, delivering a minimal, auth-less local web dashboard that queues video processing jobs, runs the vision pipeline in background worker processes, and renders interactive 3D meshes directly in the browser using Three.js.

---

## 🛠️ Key Architectural Components Completed

### 1. Vision & Spatial Reconstruction Core (`src/`)
- **Pipeline Orchestrator (`src/pipeline.py`)**: Sequential execution from video input $\rightarrow$ feature tracking $\rightarrow$ pose recovery $\rightarrow$ relative depth estimation $\rightarrow$ backprojection $\rightarrow$ point cloud fusion $\rightarrow$ surface meshing.
- **Feature Tracking (`src/tracking/feature_tracker.py`)**: ORB keypoint extraction & FLANN matching.
- **Motion Recovery (`src/tracking/pose_estimator.py`)**: Essential matrix calculation with RANSAC & chirality checks.
- **Monocular Relative Depth (`src/depth/depth_estimator.py`)**: Deep learning depth estimation via Hugging Face `Depth-Anything-V2-Small-hf`.
- **PointCloud & Mesh Builder (`src/reconstruction/`)**: Open3D voxel grid downsampling, statistical outlier removal, and Poisson surface reconstruction.
- **Telemetry Hooks**: Added `--progress-json` support to broadcast live stage progress (`extracting_frames` $\rightarrow$ `pose` $\rightarrow$ `depth` $\rightarrow$ `fusion` $\rightarrow$ `mesh` $\rightarrow$ `done`).

---

### 2. Backend & Asynchronous Job Architecture (`backend/`)
- **Flask Application Factory (`backend/app.py`)**: Clean modular routing with blueprints (`pages`, `upload`, `jobs`, `results`, `telemetry`).
- **SQLite Database (`backend/db.py` & `data/minab.db`)**: Persistent state management across server restarts.
  - `datasets`: Metadata, filename, `source_type` (`'upload'` | `'uav'`), and `has_telemetry` flag.
  - `jobs`: Queue status (`'queued'`, `'running'`, `'done'`, `'failed'`), progress percentage, error stacktraces, and processing options.
  - `results`: Links to `.ply` point clouds, `.obj` meshes, and `.txt` trajectory files with `units = 'relative'`.
  - `missions`: Pre-allocated table for Level 2+ UAV waypoint missions.
- **Background Worker (`backend/worker/`)**: Threaded daemon loop (`runner.py`) running vision tasks out-of-process via `vision_adapter.py`.
- **Automatic `.venv` Isolation**: Auto-detects the project's virtual environment to guarantee PyTorch, Torchvision, Transformers, OpenCV, and Open3D execution regardless of how Flask is launched.
- **Secure File Serving (`backend/routes/results.py`)**: Strict path traversal validation and allowlist filtering (`fused_point_cloud.ply`, `reconstructed_mesh.obj`, `camera_trajectory.txt`).

---

### 3. Frontend & Interactive 3D Web Viewer (`backend/templates/` & `backend/static/`)
- **Minab Design System**: Dark slate theme (`#090d16`) with Google Sans / Outfit typography, wireframe blue contour accents, and custom logo integration (`backend/static/logo.png`).
- **Upload & Queue Dashboard (`index.html`)**:
  - Drag-and-drop video file picker (`.mp4`, `.mov`, `.avi`).
  - Collapsible advanced controls (Frame Stride, Max Frames cap, Camera Calibration YAML).
  - Real-time auto-refreshing jobs table (polling `/api/jobs`).
- **Job Status & 3D Viewer (`job.html`)**:
  - Live progress bar polling every 1 second.
  - Interactive stage status indicators.
  - One-click job retry button for failed executions.
- **Three.js WebGL Viewer (`backend/static/js/viewer.js`)**:
  - **Orbit Controls**: Rotate, pan, and zoom in 3D space.
  - **Layer Toggles**: Surface Mesh (`.obj`), Fused Point Cloud (`.ply`), Camera Trajectory path, and Grid/Axes helpers.
  - **Scale Ambiguity Banner**: Explicitly notifies users that coordinates are in arbitrary relative units ($||t|| = 1.0$).
  - **Artifact Downloads**: Direct download links for output files.

---

### 4. Local Execution Launchers
- **`run.py` / `run_dev.py`**: One-command entry point launching Flask server and background worker on `http://localhost:5000`.

---

## 🛩️ UAV Integration Seams (Level 2+ Preparedness)

The system was designed from Day 1 with explicit extension seams so UAV features drop in without rewriting core modules:

| Seam | Level-1 Today | Level 2+ UAV Roadmap |
|------|---------------|----------------------|
| `datasets.source_type` | `'upload'` | `'uav'` for drone flight datasets |
| `datasets.telemetry.csv` | Empty stub file | Parsed MAVLink / PX4 flight log |
| `datasets.mission_id` | `NULL` | FK referencing `missions` table |
| `GET /api/telemetry/<dataset_id>` | STUB (returns `404`) | Returns flight telemetry JSON |
| `results.units` | `'relative'` | `'meters'` (georeferenced metric scale) |

---

## 📅 Roadmap & Next Steps

1. **Level 2a — UAV Telemetry Ingestion**:
   - Accept MAVLink / PX4 telemetry logs alongside video.
   - Parse camera timestamps and GPS / IMU poses into `telemetry.csv`.
   - Pass `--telemetry` to the vision pipeline to scale reconstruction to metric meters.
2. **Level 2b — Live Video Streaming**:
   - Stream UDP / RTSP drone feeds directly into `source_type = 'uav'` datasets via GStreamer.
3. **Level 3 — Mission Planner UI**:
   - Mapbox / Leaflet interactive mission planner UI to draw flight polygons into the `missions` table.
4. **Level 4 — Active Exploration Loop**:
   - Compute reconstruction uncertainty maps and auto-generate next drone mission waypoints.
