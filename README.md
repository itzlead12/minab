# Monocular 3D Reconstruction Proof-of-Concept (POC)

A modular, end-to-end Python pipeline that converts monocular video (such as a phone camera walkaround) into a calibrated camera trajectory, dense fused 3D point cloud, and surface mesh.

---

## ⚠️ Important Monocular Geometry Notice (Scale Ambiguity)

Monocular camera reconstruction from epipolar geometry (`cv2.recoverPose()`) has an inherent **scale ambiguity**:
- The translation vector between consecutive views is normalized to unit length: $\|\mathbf{t}\| = 1.0$.
- A single camera cannot determine whether a scene moved 1 centimeter, 1 meter, or 100 meters without external references (such as an IMU, stereo baseline, or known target dimensions).
- **Depth Anything V2** estimates **relative scene depth**, not guaranteed metric depth.
- **Therefore, all coordinates ($X, Y, Z$), camera trajectories, voxel grids, point clouds, and meshes in this pipeline are represented in ARBITRARY RELATIVE UNITS, NOT METRIC METERS.**

---

## Architecture

```text
                     PHONE CAMERA VIDEO
                             │
                             ▼
                    OpenCV Frame Capture
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
            Frame (k-1)              Frame (k)
                 │                       │
                 └───────────┬───────────┘
                             ▼
                    ORB Feature Tracking
                             │
                             ▼
                  Lowe's Ratio BFMatcher
                             │
                             ▼
                 Essential Matrix (RANSAC)
                             │
                             ▼
                     cv2.recoverPose()
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
       Relative Motion (R, t)    Depth Anything V2
                 │                       │
                 ▼                       ▼
       Accumulated Trajectory    Relative Depth Map
       (T_world_camera)                  │
                 │                       ▼
                 │             Pinhole Back-projection
                 │                       │
                 └───────────┬───────────┘
                             ▼
               Relative Multi-View Fusion
               (Voxel Downsample + Outlier Removal)
                             │
                             ▼
                     Dense Point Cloud (.ply)
                             │
                             ▼
                  Open3D Mesh Reconstruction
                  (Screened Poisson / BPA)
                             │
                             ▼
                   3D Mesh (.ply / .obj)
                             │
                             ▼
                  Interactive 3D Visualizer
```

---

## Prerequisites & Setup

This POC is verified on **Python 3.10** (due to official wheel availability for `open3d`, `torch`, and `opencv-python` on Windows).

```powershell
# Create virtual environment
py -3.10 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## Step-by-Step Usage

### 1. Camera Calibration (Chessboard)
Print a standard checkerboard (e.g. 9x6 inner corners, 25mm squares) or display it on a rigid tablet/monitor:
```powershell
# From photos:
python src/calibration/calibrate_camera.py --images "data/calibration/*.jpg" --pattern 9x6 --square-size 25 --output configs/camera.yaml

# Or from a calibration video:
python src/calibration/calibrate_camera.py --video "data/calibration/calib.mp4" --pattern 9x6 --output configs/camera.yaml
```
*(A default generic configuration is provided at `configs/camera_default.yaml`)*.

### 2. Capture Real Phone Video
- Place a textured, non-reflective object on a table (e.g., a chair, box, statue, textured rock, or backpack).
- Avoid reflective glass, plain white walls, or uniform mirrors.
- Walk slowly in an arc around the object (1-2 meters away), keeping the object centered in frame.
- Save the `.mp4` into `data/input_videos/my_video.mp4`.

### 3. Run Reconstruction Pipeline
```powershell
python src/pipeline.py --video data/input_videos/my_video.mp4 --camera configs/camera.yaml --config configs/pipeline_config.yaml
```

Options:
- `--frame-stride 2`: Process every 2nd frame (avoids redundant stationary frames).
- `--max-frames 30`: Cap processing to 30 keyframes for faster turnaround.
- `--headless`: Run without popping up the interactive 3D GUI window.
- `--output-dir data/output`: Target directory for exported files.

### 4. Inspect Outputs
The pipeline outputs to `data/output/`:
- `fused_point_cloud.ply`: Cleaned, voxel-downsampled multi-view 3D point cloud with RGB colors.
- `reconstructed_mesh.ply` & `reconstructed_mesh.obj`: 3D triangle surface mesh (Poisson or Ball-Pivoting).
- `camera_trajectory.txt`: Estimated camera positions and quaternion orientations in standard format.

---

## Running Software Integration Tests
A synthetic scene generator is included **strictly as a software integration test harness** to verify math and pipeline stability in CI without physical hardware:

```powershell
# 1. Generate synthetic integration test video
python tools/generate_test_sequence.py --frames 30

# 2. Run automated test suite
pytest tests/ -v

# 3. Test pipeline headlessly on the synthetic test sequence
python src/pipeline.py --video data/input_videos/synthetic_test.mp4 --camera configs/camera_synthetic.yaml --headless
```
