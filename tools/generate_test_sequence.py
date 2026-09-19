"""
Synthetic Test Sequence Generator
FOR SOFTWARE INTEGRATION & UNIT TESTING ONLY.

IMPORTANT NOTICE:
This synthetic generator exists strictly to verify software math, pipeline components,
and CI integration without needing physical hardware.
Per project requirements, real phone-camera video must be used for actual POC validation.
"""

import argparse
import math
import os
import cv2
import numpy as np
import yaml


def render_cube_scene(
    width: int = 640,
    height: int = 480,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 320.0,
    cy: float = 240.0,
    num_frames: int = 30,
    output_video_path: str = "data/input_videos/synthetic_test.mp4",
):
    """
    Renders an orbiting camera around a textured 3D cube with rich feature patterns.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # 3D vertices of a cube of side length 1.0 centered at (0, 0, 0)
    half = 0.5
    cube_vertices = np.array([
        [-half, -half, -half],
        [half, -half, -half],
        [half, half, -half],
        [-half, half, -half],
        [-half, -half, half],
        [half, -half, half],
        [half, half, half],
        [-half, half, half],
    ], dtype=np.float64)

    # 6 faces of the cube (each face has 4 vertex indices)
    faces = [
        ([0, 1, 2, 3], (255, 120, 50)),   # Back
        ([4, 5, 6, 7], (50, 120, 255)),   # Front
        ([0, 1, 5, 4], (50, 220, 50)),    # Bottom
        ([2, 3, 7, 6], (220, 220, 50)),   # Top
        ([0, 3, 7, 4], (200, 50, 200)),   # Left
        ([1, 2, 6, 5], (50, 200, 220)),   # Right
    ]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # Camera orbits at radius R around the cube at height H
    orbit_radius = 2.2
    orbit_height = 0.6
    angles = np.linspace(0, math.pi * 0.5, num_frames)

    print(f"[Synthetic Test Generator] Rendering {num_frames} frames to {output_video_path}...")

    for frame_idx, angle in enumerate(angles):
        # Camera position in world frame
        cam_x = orbit_radius * math.sin(angle)
        cam_y = -orbit_height
        cam_z = orbit_radius * math.cos(angle)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Look-at matrix: camera looks at target (0, 0, 0)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right = np.cross(world_up, forward)
        norm_r = np.linalg.norm(right)
        if norm_r > 1e-6:
            right = right / norm_r
        else:
            right = np.array([1.0, 0.0, 0.0])

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Rotation matrix from world to camera
        R_w2c = np.vstack([right, up, forward])
        t_w2c = -R_w2c @ cam_pos

        # Initialize image with textured background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (35, 35, 40) # dark gray background

        # Add background grid
        for gy in range(0, height, 40):
            cv2.line(img, (0, gy), (width, gy), (45, 45, 50), 1)
        for gx in range(0, width, 40):
            cv2.line(img, (gx, 0), (gx, height), (45, 45, 50), 1)

        # Project cube vertices to camera frame and image plane
        pts_cam = (R_w2c @ cube_vertices.T + t_w2c.reshape(3, 1)).T
        pts_2d = []
        for p in pts_cam:
            z = p[2]
            if z > 0.1:
                u = int(fx * p[0] / z + cx)
                v = int(fy * p[1] / z + cy)
                pts_2d.append((u, v))
            else:
                pts_2d.append((-100, -100))

        # Sort faces by average depth (Painter's algorithm)
        face_depths = []
        for face_indices, color in faces:
            avg_z = np.mean([pts_cam[i][2] for i in face_indices])
            face_depths.append((avg_z, face_indices, color))
        face_depths.sort(key=lambda x: x[0], reverse=True)

        # Draw faces
        for avg_z, face_indices, base_color in face_depths:
            poly = np.array([pts_2d[i] for i in face_indices], dtype=np.int32)
            cv2.fillPoly(img, [poly], base_color)
            cv2.polylines(img, [poly], True, (255, 255, 255), 2)

            # Draw textured checker pattern inside polygon to ensure strong ORB features
            center_pt = np.mean(poly, axis=0).astype(int)
            cv2.circle(img, tuple(center_pt), 6, (0, 0, 0), -1)
            cv2.circle(img, tuple(center_pt), 4, (255, 255, 255), -1)

            for corner in poly:
                cv2.circle(img, tuple(corner), 4, (0, 0, 255), -1)
                # Draw crosshairs
                cv2.line(img, (corner[0] - 8, corner[1]), (corner[0] + 8, corner[1]), (255, 255, 255), 1)
                cv2.line(img, (corner[0], corner[1] - 8), (corner[0], corner[1] + 8), (255, 255, 255), 1)

        out.write(img)

    out.release()
    print(f"[Synthetic Test Generator] Successfully created test video: {output_video_path}")

    # Also save matching camera YAML config
    synthetic_cam_config = {
        "camera_name": "synthetic_pinhole_camera_for_testing",
        "image_width": width,
        "image_height": height,
        "camera_matrix": K.tolist(),
        "dist_coeff": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam_yaml_path = "configs/camera_synthetic.yaml"
    with open(cam_yaml_path, "w") as f:
        yaml.dump(synthetic_cam_config, f)
    print(f"[Synthetic Test Generator] Saved matching camera config: {cam_yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic video sequence for software integration testing.")
    parser.add_argument("--output", type=str, default="data/input_videos/synthetic_test.mp4")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    render_cube_scene(num_frames=args.frames, output_video_path=args.output)
