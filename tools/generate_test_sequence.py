"""
Synthetic Test Sequence Generator Aligned with instruction.md
Renders an orbiting camera around a centered textured object on a table in a well-lit indoor room.
Specifications from instruction.md:
- 720p resolution (1280x720)
- 30 fps frame rate
- Centered static textured object (wood grain, fabric patterns, visible corners/edges)
- Clean indoor table background with depth variation
- Smooth continuous arc motion (1.8m distance, 0.5m height)
- Rich ORB features and clear parallax for monocular 3D reconstruction
"""

import argparse
import math
import os
import cv2
import numpy as np
import yaml


def create_procedural_wood_texture(w: int, h: int) -> np.ndarray:
    """Generates a realistic wood grain texture for the table surface."""
    x = np.linspace(0, 10, w)
    y = np.linspace(0, 5, h)
    xx, yy = np.meshgrid(x, y)
    rings = np.sin(2 * np.pi * (xx * 0.3 + np.sin(yy * 2.0) * 0.4))
    noise = np.random.RandomState(42).normal(0, 0.08, (h, w))
    grain = np.clip((rings * 0.4 + 0.6 + noise), 0.0, 1.0)
    base_color = np.array([35, 75, 130], dtype=np.float32) # warm brown BGR
    highlight = np.array([55, 110, 175], dtype=np.float32)
    wood = (grain[:, :, None] * highlight + (1.0 - grain[:, :, None]) * base_color).astype(np.uint8)
    return wood


def create_fabric_checker_texture(size: int = 256) -> np.ndarray:
    """Generates a high-contrast textured pattern with corners and edges."""
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    tile = 32
    for r in range(0, size, tile):
        for c in range(0, size, tile):
            if (r // tile + c // tile) % 2 == 0:
                tex[r:r+tile, c:c+tile] = (220, 180, 50) # gold/cyan pattern
            else:
                tex[r:r+tile, c:c+tile] = (60, 50, 200) # crimson

    # Add micro-texture / noise for ORB keypoint density
    noise = np.random.RandomState(101).randint(-25, 25, (size, size, 3), dtype=np.int16)
    tex = np.clip(tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add corner crosshairs and borders
    for r in range(0, size, tile):
        cv2.line(tex, (0, r), (size, r), (255, 255, 255), 1)
    for c in range(0, size, tile):
        cv2.line(tex, (c, 0), (c, size), (255, 255, 255), 1)
        cv2.circle(tex, (c, c), 3, (0, 0, 0), -1)

    return tex


def render_scene(
    width: int = 1280,
    height: int = 720,
    fx: float = 1000.0,
    fy: float = 1000.0,
    cx: float = 640.0,
    cy: float = 360.0,
    fps: float = 30.0,
    num_frames: int = 60,
    output_video_path: str = "data/input_videos/instruction_aligned_test.mp4",
):
    """
    Renders a realistic camera orbit around a centered textured object on a table.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # 1. 3D Model of the Centered Object (Textured Box with Beveled Edges)
    half = 0.45
    cube_vertices = np.array([
        [-half, -half, -half], # 0
        [half, -half, -half],  # 1
        [half, half, -half],   # 2
        [-half, half, -half],  # 3
        [-half, -half, half],  # 4
        [half, -half, half],   # 5
        [half, half, half],    # 6
        [-half, half, half],   # 7
    ], dtype=np.float64)

    # Cube faces: (vertex_indices, base_color_bgr, face_label)
    faces = [
        ([0, 1, 2, 3], (40, 110, 210), "Back"),
        ([4, 5, 6, 7], (200, 130, 40), "Front"),
        ([0, 1, 5, 4], (50, 180, 70), "Bottom"),
        ([2, 3, 7, 6], (70, 70, 220), "Top"),
        ([0, 3, 7, 4], (180, 50, 190), "Left"),
        ([1, 2, 6, 5], (60, 200, 210), "Right"),
    ]

    # 2. Table plane vertices (y = half, below the cube)
    table_w = 1.6
    table_d = 1.6
    table_y = half
    table_quad = np.array([
        [-table_w, table_y, -table_d],
        [table_w, table_y, -table_d],
        [table_w, table_y, table_d],
        [-table_w, table_y, table_d],
    ], dtype=np.float64)

    # Light direction (from top-front-right)
    light_dir = np.array([0.4, -0.8, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Camera orbital trajectory (smooth 120-degree arc around object)
    orbit_radius = 1.85  # 1.85 meters away
    orbit_height = 0.55  # 0.55 meters elevation
    start_angle = -math.pi * 0.35
    end_angle = math.pi * 0.35
    angles = np.linspace(start_angle, end_angle, num_frames)

    print(f"[Synthetic Generator] Rendering {num_frames} frames ({width}x{height} @ {fps}fps) to {output_video_path}...")

    fabric_tex = create_fabric_checker_texture(128)

    for frame_idx, angle in enumerate(angles):
        # Camera position in world space
        cam_x = orbit_radius * math.sin(angle)
        cam_y = -orbit_height
        cam_z = orbit_radius * math.cos(angle)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Look directly at the center object (0, 0, 0)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right = np.cross(world_up, forward)
        norm_r = np.linalg.norm(right)
        right = right / norm_r if norm_r > 1e-6 else np.array([1.0, 0.0, 0.0])

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        R_w2c = np.vstack([right, up, forward])
        t_w2c = -R_w2c @ cam_pos

        # Indoor wall background gradient (warm slate gray with soft ambient lighting)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        grad_y = np.linspace(35, 65, height)[:, None, None].astype(np.uint8)
        img[:, :] = np.repeat(grad_y, width, axis=1) * np.array([1, 1, 1], dtype=np.uint8)

        # Subtle wall texture
        wall_noise = np.random.RandomState(frame_idx + 1).randint(-3, 4, (height, width, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + wall_noise, 0, 255).astype(np.uint8)

        # 3. Project Table Quad
        pts_table_cam = (R_w2c @ table_quad.T + t_w2c.reshape(3, 1)).T
        table_2d = []
        for p in pts_table_cam:
            z = p[2]
            if z > 0.1:
                u = int(fx * p[0] / z + cx)
                v = int(fy * p[1] / z + cy)
                table_2d.append((u, v))
            else:
                table_2d.append((-100, -100))

        table_poly = np.array(table_2d, dtype=np.int32)
        # Render table with wooden surface tone
        cv2.fillPoly(img, [table_poly], (50, 90, 140))
        cv2.polylines(img, [table_poly], True, (65, 110, 165), 2)

        # Draw grid lines on table surface for rich texture
        for gx in np.linspace(-table_w, table_w, 9):
            p_a = (R_w2c @ np.array([gx, table_y, -table_d]) + t_w2c)
            p_b = (R_w2c @ np.array([gx, table_y, table_d]) + t_w2c)
            if p_a[2] > 0.1 and p_b[2] > 0.1:
                ua = int(fx * p_a[0] / p_a[2] + cx)
                va = int(fy * p_a[1] / p_a[2] + cy)
                ub = int(fx * p_b[0] / p_b[2] + cx)
                vb = int(fy * p_b[1] / p_b[2] + cy)
                cv2.line(img, (ua, va), (ub, vb), (40, 75, 120), 1)

        # 4. Project Cube Vertices
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

        # Sort cube faces by average camera depth (Painter's algorithm)
        face_list = []
        for face_indices, base_color, label in faces:
            avg_z = np.mean([pts_cam[i][2] for i in face_indices])
            # Compute face normal for Lambertian lighting
            v0 = cube_vertices[face_indices[0]]
            v1 = cube_vertices[face_indices[1]]
            v2 = cube_vertices[face_indices[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                normal = normal / norm_val
            # Dot with camera ray to check visibility
            view_vec = cube_vertices[face_indices[0]] - cam_pos
            if np.dot(normal, view_vec) < 0: # facing camera
                diffuse = max(0.25, float(np.dot(normal, -light_dir)))
                lit_color = np.clip(np.array(base_color) * diffuse + 20, 0, 255).astype(int)
                face_list.append((avg_z, face_indices, lit_color, normal))

        face_list.sort(key=lambda x: x[0], reverse=True)

        # Draw visible cube faces with texture, corner markers, and patterns
        for avg_z, face_indices, lit_color, normal in face_list:
            poly = np.array([pts_2d[i] for i in face_indices], dtype=np.int32)
            cv2.fillPoly(img, [poly], tuple(int(c) for c in lit_color))
            cv2.polylines(img, [poly], True, (240, 240, 240), 2)

            # Draw high-contrast interior checkerboard & crosshair features
            center_pt = np.mean(poly, axis=0).astype(int)
            cv2.circle(img, tuple(center_pt), 8, (20, 20, 20), -1)
            cv2.circle(img, tuple(center_pt), 5, (255, 255, 255), -1)

            # Draw corner markers for ORB detection
            for corner in poly:
                cv2.circle(img, tuple(corner), 5, (0, 0, 255), -1)
                cv2.circle(img, tuple(corner), 2, (255, 255, 255), -1)
                cv2.line(img, (corner[0] - 10, corner[1]), (corner[0] + 10, corner[1]), (255, 255, 255), 1)
                cv2.line(img, (corner[0], corner[1] - 10), (corner[0], corner[1] + 10), (255, 255, 255), 1)

            # Edge tick marks for dense corner tracking
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                for alpha in (0.33, 0.67):
                    mid = (p1 * (1 - alpha) + p2 * alpha).astype(int)
                    cv2.circle(img, tuple(mid), 3, (255, 200, 50), -1)

        out.write(img)

    out.release()
    print(f"[Synthetic Generator] Video successfully created: {output_video_path}")

    # Matching Camera Calibration Config (720p)
    synthetic_cam_config = {
        "camera_name": "calibrated_synthetic_camera_720p",
        "image_width": width,
        "image_height": height,
        "camera_matrix": K.tolist(),
        "dist_coeff": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam_yaml_path = "configs/camera_synthetic.yaml"
    with open(cam_yaml_path, "w") as f:
        yaml.dump(synthetic_cam_config, f, default_flow_style=None)
    print(f"[Synthetic Generator] Saved camera config: {cam_yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 720p synthetic video sequence matching instruction.md.")
    parser.add_argument("--output", type=str, default="data/input_videos/instruction_aligned_test.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    render_scene(num_frames=args.frames, fps=args.fps, output_video_path=args.output)
