"""Unit tests for point cloud fusion and Open3D surface mesh generation."""

import numpy as np
import open3d as o3d
import pytest
from src.reconstruction.cloud_fusion import PointCloudFusion
from src.reconstruction.mesh_builder import MeshBuilder


def test_point_cloud_fusion_transformation():
    fusion = PointCloudFusion(voxel_size=0.01)

    # Frame 1: point at (0, 0, 1) in camera frame, camera at origin
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]))
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))

    T_w_c1 = np.eye(4)
    fusion.add_frame_cloud(pcd1, T_w_c1)

    # Frame 2: point at (0, 0, 1) in camera frame, camera translated by X=+0.5 in world
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]]))
    pcd2.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))

    T_w_c2 = np.eye(4)
    T_w_c2[0, 3] = 0.5 # shift X by 0.5
    fusion.add_frame_cloud(pcd2, T_w_c2)

    fused = fusion.get_cloud()
    pts = np.asarray(fused.points)

    assert len(pts) == 2
    # One point should be at (0, 0, 1) and the other at (0.5, 0, 1)
    xs = sorted(pts[:, 0])
    assert np.isclose(xs[0], 0.0, atol=1e-4)
    assert np.isclose(xs[1], 0.5, atol=1e-4)


def test_mesh_builder_poisson_reconstruction():
    # Generate points sampled from a sphere of radius 1.0
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    pcd = mesh_sphere.sample_points_uniformly(number_of_points=1000)

    builder = MeshBuilder(method="poisson", poisson_depth=6, trim_density_percentile=0.01)
    mesh, densities = builder.build_mesh(pcd)

    assert isinstance(mesh, o3d.geometry.TriangleMesh)
    assert len(mesh.vertices) > 50, "Should reconstruct valid vertices"
    assert len(mesh.triangles) > 50, "Should reconstruct valid triangles"
