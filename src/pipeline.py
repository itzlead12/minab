"""
End-to-End Monocular 3D Reconstruction Pipeline
Orchestrates: Video -> Calibration -> ORB Tracking -> Essential Matrix Pose Recovery ->
              Depth Anything V2 Relative Depth -> Point Cloud Backprojection & Fusion -> Open3D Surface Mesh.

CRITICAL SCALE AMBIGUITY NOTICE:
Single-camera monocular reconstruction cannot determine physical scale without external metric sensors.
The translation vector between frames is normalized (||t|| = 1.0).
All camera trajectories, voxel grids, point clouds, and meshes are in ARBITRARY RELATIVE UNITS, NOT meters.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

# Ensure modules in src/ and root are importable regardless of working directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from calibration.calibrate_camera import load_calibration, scale_camera_matrix
from calibration.input_validator import VideoInputAssessor
from depth.depth_estimator import DepthEstimator
from reconstruction.backprojector import Backprojector
from reconstruction.cloud_fusion import PointCloudFusion
from reconstruction.mesh_builder import MeshBuilder
from tracking.feature_tracker import FeatureTracker
from tracking.pose_estimator import PoseEstimator
from visualization.visualizer_3d import Visualizer3D


def load_yaml(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_trajectory_tum_format(trajectory, output_path: str):
    """
    Saves camera poses in standard TUM/EuRoC trajectory format:
    timestamp/frame_idx tx ty tz qx qy qz qw
    """
    with open(output_path, "w") as f:
        f.write("# Monocular camera trajectory (Relative arbitrary units, NOT metric meters)\n")
        f.write("# frame_idx tx ty tz qx qy qz qw\n")
        for pose in trajectory:
            t = pose.camera_center
            R = pose.rotation
            # Convert 3x3 rotation matrix to quaternion
            # OpenCV provides Rodrigues vector or we use standard conversion
            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

            f.write(f"{pose.frame_idx} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def save_trajectory_json_format(trajectory, output_path: str):
    """
    Saves camera poses in structured JSON format with positions and quaternions
    for web 3D visualizers.
    """
    poses = []
    for pose in trajectory:
        t = pose.camera_center
        R = pose.rotation
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        poses.append({
            "frame_idx": pose.frame_idx,
            "position": [round(float(t[0]), 6), round(float(t[1]), 6), round(float(t[2]), 6)],
            "quaternion": [round(float(qx), 6), round(float(qy), 6), round(float(qz), 6), round(float(qw), 6)],
            "inliers": int(pose.inliers_count),
        })

    data = {
        "units": "relative",
        "scale_notice": "Normalized translation vector (||t||=1.0). Coordinates in arbitrary units, not meters.",
        "count": len(poses),
        "poses": poses,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


import json


def report_progress(progress_json_path: Optional[str], stage: str, progress: float):
    if not progress_json_path:
        return
    try:
        tmp_path = progress_json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"stage": stage, "progress": round(progress, 3)}, f)
        os.replace(tmp_path, progress_json_path)
    except Exception:
        pass


def run_pipeline(
    video_path: str,
    camera_config_path: str,
    pipeline_config_path: str,
    output_dir: str = "data/output",
    max_frames: Optional[int] = None,
    frame_stride: int = 2,
    headless: bool = False,
    progress_json: Optional[str] = None,
):
    print("=" * 70)
    print("  MONOCULAR 3D RECONSTRUCTION PROOF OF CONCEPT")
    print("=" * 70)
    print("  SCALE NOTICE: All coordinates and trajectory points are in")
    print("  ARBITRARY RELATIVE UNITS due to monocular scale ambiguity (||t||=1).")
    print("  Do NOT interpret outputs as physical metric measurements.")
    report_progress(progress_json, "extracting_frames", 0.05)
    print("=" * 70)

    # 1. Load configurations and assess video
    os.makedirs(output_dir, exist_ok=True)
    cam_params = load_calibration(camera_config_path)
    pipe_cfg = load_yaml(pipeline_config_path)

    # Validate input video against instruction.md
    print("[Pipeline] Assessing video input quality against instruction.md...")
    try:
        assessor = VideoInputAssessor()
        assessment = assessor.assess_video(video_path)
        assessment_out_path = os.path.join(output_dir, "input_assessment.json")
        with open(assessment_out_path, "w") as f:
            json.dump(assessment, f, indent=2)
        print(f"[Pipeline] Video Assessment: {assessment['resolution_quality']}, {assessment['fps']}fps, Sharpness: {assessment['sharpness_grade']} ({assessment['sharpness_score']}), Motion: {assessment['motion_type']}")
        for w in assessment.get("warnings", []):
            print(f"  [Input Warning] {w}")
    except Exception as ex:
        print(f"[Pipeline] Warning: Could not run input assessment: {ex}")

    # Open video to check dimensions and scale K if needed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calib_w = cam_params.get("image_width", video_w)
    calib_h = cam_params.get("image_height", video_h)

    K = cam_params["camera_matrix"]
    D = cam_params["dist_coeff"]

    if video_w != calib_w or video_h != calib_h:
        print(f"[Pipeline] Video resolution ({video_w}x{video_h}) differs from calibration ({calib_w}x{calib_h}). Dynamically scaling intrinsics K.")
        K = scale_camera_matrix(K, calib_w, calib_h, video_w, video_h)

    print(f"[Pipeline] Loaded camera intrinsics K:\n{K}")

    # 2. Initialize modules
    tracker_cfg = pipe_cfg.get("feature_tracker", {})
    tracker = FeatureTracker(
        max_features=tracker_cfg.get("max_features", 2000),
        scale_factor=tracker_cfg.get("scale_factor", 1.2),
        n_levels=tracker_cfg.get("n_levels", 8),
        match_ratio=tracker_cfg.get("match_ratio", 0.75),
        min_matches=tracker_cfg.get("min_matches", 25),
    )

    pose_cfg = pipe_cfg.get("pose_estimator", {})
    pose_estimator = PoseEstimator(
        camera_matrix=K,
        dist_coeff=D,
        ransac_prob=pose_cfg.get("ransac_prob", 0.999),
        ransac_threshold_px=pose_cfg.get("ransac_threshold_px", 1.2),
        min_inliers=pose_cfg.get("min_inliers", 20),
    )

    depth_cfg = pipe_cfg.get("depth_estimator", {})
    depth_estimator = DepthEstimator(
        model_id=depth_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf"),
        device=depth_cfg.get("device", "auto"),
        relative_depth_min=depth_cfg.get("relative_depth_min", 0.5),
        relative_depth_max=depth_cfg.get("relative_depth_max", 5.0),
        max_inference_dimension=depth_cfg.get("max_inference_dimension", 518),
    )

    recon_cfg = pipe_cfg.get("reconstruction", {})
    backproj_cfg = recon_cfg.get("backprojection", {})
    backprojector = Backprojector(
        camera_matrix=K,
        pixel_stride=backproj_cfg.get("pixel_stride", 2),
        relative_depth_trunc=backproj_cfg.get("relative_depth_trunc", 8.0),
    )

    fusion_cfg = recon_cfg.get("fusion", {})
    fusion = PointCloudFusion(
        voxel_size=fusion_cfg.get("voxel_size", 0.02),
        outlier_nb_neighbors=fusion_cfg.get("outlier_nb_neighbors", 20),
        outlier_std_ratio=fusion_cfg.get("outlier_std_ratio", 2.0),
    )

    mesh_cfg = recon_cfg.get("meshing", {})
    mesh_builder = MeshBuilder(
        method=mesh_cfg.get("method", "poisson"),
        poisson_depth=mesh_cfg.get("poisson_depth", 9),
        trim_density_percentile=mesh_cfg.get("trim_density_percentile", 0.05),
        bpa_radii=mesh_cfg.get("bpa_radii", [0.02, 0.04, 0.08]),
    )

    print(f"[Pipeline] Processing video: {video_path} ({total_video_frames} total frames, {video_w}x{video_h})")

    raw_frame_idx = 0
    processed_count = 0

    prev_frame_bgr = None
    prev_kp = None
    prev_des = None

    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if raw_frame_idx % frame_stride != 0:
            raw_frame_idx += 1
            continue

        if max_frames is not None and processed_count >= max_frames:
            print(f"[Pipeline] Reached maximum requested frame limit: {max_frames}")
            break

        print(f"\n--- [Frame {raw_frame_idx}] (Processed #{processed_count + 1}) ---")
        kp, des = tracker.extract(frame_bgr)
        print(f"  Detected {len(kp)} ORB features.")

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if processed_count == 0:
            # First keyframe at origin T_w_0 = I
            depth_map = depth_estimator.estimate(frame_bgr, sharpen_edges=True)
            local_cloud = backprojector.backproject(frame_bgr, depth_map)
            print(f"  Frame 0: Backprojected {len(local_cloud.points)} initial points.")
            fusion.add_frame_cloud(local_cloud, pose_estimator.current_T_world_camera)

            prev_frame_bgr = frame_bgr
            prev_gray = curr_gray
            prev_kp = kp
            prev_des = des
            processed_count += 1
            raw_frame_idx += 1
            continue

        # Match with previous keyframe using sub-pixel refinement
        match_res = tracker.match(
            prev_kp, prev_des, kp, des,
            frame1_gray=prev_gray, frame2_gray=curr_gray
        )
        if match_res is None:
            print("  [Warning] Insufficient feature correspondences with previous keyframe. Skipping frame.")
            raw_frame_idx += 1
            continue

        print(f"  Matched {len(match_res.matches)} features with previous keyframe.")

        # Estimate camera motion
        pose = pose_estimator.update_pose(
            frame_idx=raw_frame_idx,
            pts1=match_res.pts1,
            pts2=match_res.pts2,
            is_keyframe=True,
        )

        if pose is None:
            print("  [Warning] Pose recovery failed chirality or inlier threshold. Skipping frame.")
            raw_frame_idx += 1
            continue

        cam_pos = pose.camera_center
        print(f"  Camera Pose #{len(pose_estimator.trajectory)}: Pos=[{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}] (Inliers: {pose.inliers_count})")

        # Estimate depth map with edge sharpening and backproject
        depth_map = depth_estimator.estimate(frame_bgr, sharpen_edges=True)
        local_cloud = backprojector.backproject(frame_bgr, depth_map)
        print(f"  Backprojected {len(local_cloud.points)} 3D points.")

        # Fuse into global model using accumulated camera pose
        fusion.add_frame_cloud(local_cloud, pose.transform_matrix)
        print(f"  Current fused cloud size: {len(fusion.get_cloud().points)} points.")

        # Advance keyframe
        prev_frame_bgr = frame_bgr
        prev_gray = curr_gray
        prev_kp = kp
        prev_des = des
        processed_count += 1
        raw_frame_idx += 1

        pct = 0.10 + min(0.65, (raw_frame_idx / max(total_video_frames, 1)) * 0.65)
        report_progress(progress_json, "depth", pct)

    cap.release()
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[Pipeline] Video processing finished in {elapsed:.2f}s ({processed_count} keyframes processed)")

    # 4. Filter and save fused point cloud
    report_progress(progress_json, "fusion", 0.80)
    print("[Pipeline] Filtering point cloud outliers and estimating surface normals...")
    fusion.filter_outliers(enable_radius_filter=True)
    fusion.downsample()
    fusion.estimate_normals()
    fused_pcd = fusion.get_cloud()
    pcd_out_path = os.path.join(output_dir, "fused_point_cloud.ply")
    fusion.save(pcd_out_path)
    print(f"[Pipeline] Saved fused point cloud ({len(fused_pcd.points)} points) to: {pcd_out_path}")

    # 5. Save trajectory (TUM text format + JSON for Three.js web viewer)
    traj_out_path = os.path.join(output_dir, "camera_trajectory.txt")
    save_trajectory_tum_format(pose_estimator.trajectory, traj_out_path)
    print(f"[Pipeline] Saved camera trajectory to: {traj_out_path}")

    traj_json_path = os.path.join(output_dir, "camera_trajectory.json")
    save_trajectory_json_format(pose_estimator.trajectory, traj_json_path)
    print(f"[Pipeline] Saved camera trajectory JSON to: {traj_json_path}")

    # 6. Reconstruct Surface Mesh
    report_progress(progress_json, "mesh", 0.90)
    mesh = None
    if len(fused_pcd.points) >= 50:
        try:
            print("[Pipeline] Reconstructing 3D surface mesh...")
            mesh, _ = mesh_builder.build_mesh(fused_pcd)
            mesh_ply_path = os.path.join(output_dir, "reconstructed_mesh.ply")
            mesh_obj_path = os.path.join(output_dir, "reconstructed_mesh.obj")
            mesh_builder.save(mesh, mesh_ply_path)
            mesh_builder.save(mesh, mesh_obj_path)
            print(f"[Pipeline] Saved mesh to: {mesh_ply_path} and {mesh_obj_path}")
        except Exception as e:
            print(f"[Pipeline] Mesh reconstruction warning: {e}")
    else:
        print("[Pipeline] Insufficient points to build mesh.")

    # 7. Visualization
    if not headless:
        print("[Pipeline] Launching Open3D interactive 3D visualizer...")
        vis_cfg = pipe_cfg.get("visualization", {})
        visualizer = Visualizer3D(window_name="Monocular 3D Reconstruction & Camera Trajectory")
        visualizer.display(
            point_cloud=fused_pcd,
            mesh=mesh,
            trajectory_points=pose_estimator.get_trajectory_points(),
            camera_poses=[p.transform_matrix for p in pose_estimator.trajectory],
            frustum_scale=vis_cfg.get("frustum_scale", 0.15),
        )

    report_progress(progress_json, "done", 1.0)
    print("[Pipeline] Pipeline execution complete!")


def main():
    parser = argparse.ArgumentParser(description="Monocular 3D Reconstruction Proof-of-Concept Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (e.g. phone walkaround)")
    parser.add_argument("--camera", type=str, default="configs/camera_default.yaml", help="Path to camera calibration YAML")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory to save reconstruction outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth video frame")
    parser.add_argument("--headless", action="store_true", help="Run without opening interactive 3D GUI window")
    parser.add_argument("--progress-json", type=str, default=None, help="Path to write progress status JSON")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        camera_config_path=args.camera,
        pipeline_config_path=args.config,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        headless=args.headless,
        progress_json=args.progress_json,
    )


if __name__ == "__main__":
    main()
