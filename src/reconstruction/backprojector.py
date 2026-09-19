"""
Backprojection Module
Converts a 2D RGB image and 2D relative depth map into a 3D Open3D Point Cloud using camera intrinsics.

Coordinate and Scale Notice:
Backprojected (X, Y, Z) coordinates reflect relative camera-space coordinates.
Since depth values are from monocular relative depth estimation, coordinates are in
arbitrary relative units, NOT metric meters.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import open3d as o3d


class Backprojector:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        pixel_stride: int = 2,
        relative_depth_trunc: float = 8.0,
        min_depth: float = 0.1,
    ):
        """
        Initializes the pinhole backprojector.

        Args:
            camera_matrix: 3x3 intrinsic matrix K with (fx, fy, cx, cy).
            pixel_stride: Pixel downsampling stride (e.g. 2 means sample every 2nd row and col).
            relative_depth_trunc: Maximum relative depth threshold for backprojection.
            min_depth: Minimum valid relative depth threshold.
        """
        self.K = np.array(camera_matrix, dtype=np.float64)
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.pixel_stride = max(1, pixel_stride)
        self.relative_depth_trunc = relative_depth_trunc
        self.min_depth = min_depth

    def backproject(
        self,
        rgb_image: np.ndarray,
        relative_depth_map: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Backprojects an RGB frame and relative depth map into an Open3D PointCloud.

        Args:
            rgb_image: HxWx3 uint8 image (BGR or RGB).
            relative_depth_map: HxW float32 relative depth map.

        Returns:
            Open3D PointCloud in relative camera coordinates.
        """
        h, w = relative_depth_map.shape[:2]

        # Convert to RGB if 3 channels BGR
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            # Check if likely BGR (standard OpenCV)
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_image

        # Create 2D pixel coordinate grid with step = pixel_stride
        u_coords = np.arange(0, w, self.pixel_stride)
        v_coords = np.arange(0, h, self.pixel_stride)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # Subsample depth and colors
        z_sub = relative_depth_map[v_grid, u_grid]
        rgb_sub = rgb[v_grid, u_grid].astype(np.float64) / 255.0

        # Mask valid depth values
        valid_mask = (z_sub > self.min_depth) & (z_sub <= self.relative_depth_trunc) & np.isfinite(z_sub)

        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        z_valid = z_sub[valid_mask]
        colors_valid = rgb_sub[valid_mask]

        # Pinhole projection math:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x_valid = (u_valid - self.cx) * z_valid / self.fx
        y_valid = (v_valid - self.cy) * z_valid / self.fy

        xyz = np.stack((x_valid, y_valid, z_valid), axis=-1).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        if len(xyz) > 0:
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors_valid)

        return pcd
