"""
Multi-View Point Cloud Fusion Module
Transforms camera-frame point clouds into the global relative world frame using estimated camera poses,
accumulates points across views, and filters noise via voxel downsampling and statistical outlier removal.

SCALE DISCLAIMER:
Fused point clouds reside in an unscaled relative coordinate system inherited from monocular recoverPose().
Coordinates are NOT in meters.
"""

from typing import Optional
import numpy as np
import open3d as o3d


class PointCloudFusion:
    def __init__(
        self,
        voxel_size: float = 0.02,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
    ):
        """
        Initializes point cloud accumulator and filter.

        Args:
            voxel_size: Voxel downsample grid size (in relative units).
            outlier_nb_neighbors: Number of neighbors to analyze for statistical outlier removal.
            outlier_std_ratio: Standard deviation threshold ratio for outlier removal.
        """
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        self.fused_cloud = o3d.geometry.PointCloud()
        self.frames_fused_count = 0

    def reset(self):
        """Resets the fused point cloud."""
        self.fused_cloud = o3d.geometry.PointCloud()
        self.frames_fused_count = 0

    def add_frame_cloud(
        self,
        local_pcd: o3d.geometry.PointCloud,
        T_world_camera: np.ndarray,
        downsample_on_add: bool = True,
    ) -> None:
        """
        Transforms a local camera-frame point cloud into the global relative frame and merges it.

        Args:
            local_pcd: Open3D PointCloud in camera optical frame.
            T_world_camera: 4x4 matrix mapping camera frame to world frame (P_world = T_w_c @ P_camera).
            downsample_on_add: If True, performs voxel downsampling to keep memory footprint bounded.
        """
        if len(local_pcd.points) == 0:
            return

        # Clone and transform into global relative space
        cloud_transformed = o3d.geometry.PointCloud(local_pcd)
        cloud_transformed.transform(T_world_camera)

        # Merge with global cloud
        self.fused_cloud += cloud_transformed
        self.frames_fused_count += 1

        if downsample_on_add and self.frames_fused_count % 3 == 0:
            self.downsample()

    def downsample(self) -> None:
        """Downsamples the fused point cloud with a regular 3D voxel grid."""
        if len(self.fused_cloud.points) > 0 and self.voxel_size > 0:
            self.fused_cloud = self.fused_cloud.voxel_down_sample(voxel_size=self.voxel_size)

    def filter_outliers(self, enable_radius_filter: bool = True) -> o3d.geometry.PointCloud:
        """
        Removes isolated noise and floating points using dual statistical and radius outlier removal.

        Returns:
            Filtered Open3D PointCloud.
        """
        if len(self.fused_cloud.points) < self.outlier_nb_neighbors:
            return self.fused_cloud

        # 1. Statistical outlier removal for density-based noise
        cl, _ = self.fused_cloud.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio,
        )
        self.fused_cloud = cl

        # 2. Radius outlier removal for floating depth artifacts
        if enable_radius_filter and len(self.fused_cloud.points) > 50:
            radius = max(0.05, self.voxel_size * 3.5)
            min_pts = max(6, int(self.outlier_nb_neighbors * 0.4))
            cl_rad, _ = self.fused_cloud.remove_radius_outlier(
                nb_points=min_pts,
                radius=radius,
            )
            self.fused_cloud = cl_rad

        return self.fused_cloud

    def estimate_normals(self, max_nn: int = 30) -> None:
        """Estimates and orientates surface normals for the fused point cloud."""
        if len(self.fused_cloud.points) < 10:
            return
        radius = max(0.04, self.voxel_size * 2.5)
        self.fused_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        self.fused_cloud.orient_normals_consistent_tangent_plane(k=15)

    def get_cloud(self) -> o3d.geometry.PointCloud:
        """Returns the current fused point cloud."""
        return self.fused_cloud

    def save(self, filepath: str) -> bool:
        """Saves the fused point cloud to a .ply or .pcd file."""
        return o3d.io.write_point_cloud(filepath, self.fused_cloud)
