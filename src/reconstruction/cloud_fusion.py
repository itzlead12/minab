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

    def filter_outliers(self) -> o3d.geometry.PointCloud:
        """
        Removes isolated noise points using statistical outlier removal.

        Returns:
            Filtered Open3D PointCloud.
        """
        if len(self.fused_cloud.points) < self.outlier_nb_neighbors:
            return self.fused_cloud

        cl, ind = self.fused_cloud.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio,
        )
        self.fused_cloud = cl
        return self.fused_cloud

    def get_cloud(self) -> o3d.geometry.PointCloud:
        """Returns the current fused point cloud."""
        return self.fused_cloud

    def save(self, filepath: str) -> bool:
        """Saves the fused point cloud to a .ply or .pcd file."""
        return o3d.io.write_point_cloud(filepath, self.fused_cloud)
