"""
3D Visualization Module
Renders camera trajectory paths, camera frustum pyramids, point clouds, and reconstructed meshes using Open3D.
"""

from typing import List, Optional
import numpy as np
import open3d as o3d


def create_camera_frustum(
    T_world_camera: np.ndarray,
    color: List[float] = [0.0, 0.8, 0.2],
    scale: float = 0.15,
) -> o3d.geometry.LineSet:
    """
    Constructs a 3D wireframe pyramid representing a camera frustum at a given world pose.

    Args:
        T_world_camera: 4x4 camera-to-world transformation matrix.
        color: RGB color in [0, 1].
        scale: Visual size of the frustum in relative scene units.

    Returns:
        Open3D LineSet geometry.
    """
    # Define camera pyramid vertices in local camera coordinate frame
    # Optical center is at (0, 0, 0), viewing direction is +Z
    w = scale * 0.7
    h = scale * 0.5
    z = scale

    local_pts = np.array([
        [0, 0, 0],         # 0: optical center
        [-w, -h, z],       # 1: top-left
        [w, -h, z],        # 2: top-right
        [w, h, z],         # 3: bottom-right
        [-w, h, z],        # 4: bottom-left
    ], dtype=np.float64)

    # Transform to world coordinates: P_w = T_w_c @ [P_c; 1]
    ones = np.ones((local_pts.shape[0], 1))
    pts_homo = np.hstack([local_pts, ones])
    world_pts = (T_world_camera @ pts_homo.T).T[:, :3]

    # Lines connecting the vertices
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], # rays from camera center to image plane corners
        [1, 2], [2, 3], [3, 4], [4, 1], # image plane perimeter
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_trajectory_lines(
    trajectory_points: np.ndarray,
    color: List[float] = [1.0, 0.2, 0.2],
) -> Optional[o3d.geometry.LineSet]:
    """
    Creates a polyline connecting sequential camera positions.

    Args:
        trajectory_points: Nx3 array of camera optical centers.
        color: Line color in RGB [0, 1].

    Returns:
        Open3D LineSet or None if less than 2 points.
    """
    if len(trajectory_points) < 2:
        return None

    lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_ground_grid(
    center_y: float = 0.5,
    size: float = 3.0,
    n_lines: int = 16,
    color: List[float] = [0.2, 0.28, 0.38],
) -> o3d.geometry.LineSet:
    """
    Creates a studio reference grid beneath the reconstructed object.
    """
    half = size / 2.0
    step = size / n_lines
    pts = []
    lines = []

    pt_idx = 0
    # X lines
    for i in range(n_lines + 1):
        x = -half + i * step
        pts.append([x, center_y, -half])
        pts.append([x, center_y, half])
        lines.append([pt_idx, pt_idx + 1])
        pt_idx += 2

    # Z lines
    for j in range(n_lines + 1):
        z = -half + j * step
        pts.append([-half, center_y, z])
        pts.append([half, center_y, z])
        lines.append([pt_idx, pt_idx + 1])
        pt_idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


class Visualizer3D:
    def __init__(self, window_name: str = "3D Reconstruction & Camera Trajectory"):
        self.window_name = window_name

    def display(
        self,
        point_cloud: Optional[o3d.geometry.PointCloud] = None,
        mesh: Optional[o3d.geometry.TriangleMesh] = None,
        trajectory_points: Optional[np.ndarray] = None,
        camera_poses: Optional[List[np.ndarray]] = None,
        frustum_scale: float = 0.15,
        frustum_stride: int = 2,
    ) -> None:
        """
        Spawns an interactive Open3D window displaying reconstructed 3D geometry and camera trajectory.
        """
        geometries = []

        # Add world coordinate frame at origin (Red=X, Green=Y, Blue=Z)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        geometries.append(coord_frame)

        # Ground reference grid
        ground_y = 0.5
        if point_cloud is not None and len(point_cloud.points) > 0:
            pts_np = np.asarray(point_cloud.points)
            ground_y = float(np.percentile(pts_np[:, 1], 95))
        grid = create_ground_grid(center_y=ground_y, size=4.0, n_lines=20)
        geometries.append(grid)

        # Add camera trajectory polyline
        if trajectory_points is not None and len(trajectory_points) >= 2:
            traj_lines = create_trajectory_lines(trajectory_points, color=[1.0, 0.1, 0.1])
            if traj_lines is not None:
                geometries.append(traj_lines)

        # Add camera frustums along the arc
        if camera_poses is not None:
            for idx, T in enumerate(camera_poses):
                if idx % frustum_stride == 0 or idx == len(camera_poses) - 1:
                    # Initial camera green, intermediate cyan, last yellow
                    if idx == 0:
                        c = [0.0, 1.0, 0.0]
                    elif idx == len(camera_poses) - 1:
                        c = [1.0, 0.9, 0.0]
                    else:
                        c = [0.0, 0.8, 1.0]
                    frustum = create_camera_frustum(T, color=c, scale=frustum_scale)
                    geometries.append(frustum)

        # Add point cloud if present
        if point_cloud is not None and len(point_cloud.points) > 0:
            geometries.append(point_cloud)

        # Add mesh if present
        if mesh is not None and len(mesh.triangles) > 0:
            geometries.append(mesh)

        if not geometries:
            print("[Visualizer3D] Warning: No geometries to display.")
            return

        print(f"[Visualizer3D] Opening visualizer window with {len(geometries)} scene objects...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=self.window_name,
            width=1280,
            height=800,
            left=50,
            top=50,
        )
