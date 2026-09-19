"""
Surface Mesh Reconstruction Module
Builds 3D triangle surfaces from fused point clouds using Screened Poisson Reconstruction or
the Ball-Pivoting Algorithm (BPA) via Open3D.
"""

from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d


class MeshBuilder:
    def __init__(
        self,
        method: str = "poisson",
        poisson_depth: int = 8,
        trim_density_percentile: float = 0.05,
        bpa_radii: Optional[List[float]] = None,
    ):
        """
        Initializes the surface mesh builder.

        Args:
            method: 'poisson' or 'ball_pivoting'.
            poisson_depth: Octree depth for Poisson surface reconstruction (higher = finer detail, slower).
            trim_density_percentile: Percentile of low-density vertices to prune from Poisson reconstruction
                                     to eliminate extrapolation bubbles.
            bpa_radii: List of ball radii (in relative units) for Ball-Pivoting Algorithm.
        """
        self.method = method.lower()
        self.poisson_depth = poisson_depth
        self.trim_density_percentile = trim_density_percentile
        self.bpa_radii = bpa_radii or [0.02, 0.04, 0.08]

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: Optional[float] = None,
        max_nn: int = 30,
    ) -> None:
        """Estimates and orients surface normals for a point cloud."""
        if radius is None:
            # Derive radius from bounding box size if not specified
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = np.linalg.norm(bbox.get_extent())
            radius = extent * 0.03

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    def build_mesh(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> Tuple[o3d.geometry.TriangleMesh, Optional[np.ndarray]]:
        """
        Reconstructs a triangle mesh from an Open3D point cloud.

        Returns:
            Tuple of (TriangleMesh, densities_array or None).
        """
        if len(pcd.points) < 50:
            raise ValueError(f"Too few points for mesh reconstruction: {len(pcd.points)} points found.")

        # Ensure normals exist
        if not pcd.has_normals():
            print("[MeshBuilder] Point cloud lacks surface normals. Estimating normals...")
            self.estimate_normals(pcd)

        if self.method == "poisson":
            print(f"[MeshBuilder] Running Poisson Surface Reconstruction (octree depth={self.poisson_depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth
            )
            densities_np = np.asarray(densities)

            # Trim low-density extrapolation vertices
            if self.trim_density_percentile > 0:
                thresh = np.percentile(densities_np, self.trim_density_percentile * 100)
                vertices_to_remove = densities_np < thresh
                mesh.remove_vertices_by_mask(vertices_to_remove)
                print(f"[MeshBuilder] Pruned low-density Poisson vertices (< {thresh:.2f}).")

        elif self.method == "ball_pivoting":
            print(f"[MeshBuilder] Running Ball-Pivoting Algorithm with radii: {self.bpa_radii}...")
            radii_vector = o3d.utility.DoubleVector(self.bpa_radii)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii_vector
            )
            densities_np = None
        else:
            raise ValueError(f"Unknown meshing method: '{self.method}'. Expected 'poisson' or 'ball_pivoting'.")

        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        print(f"[MeshBuilder] Mesh constructed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
        return mesh, densities_np

    @staticmethod
    def save(mesh: o3d.geometry.TriangleMesh, filepath: str) -> bool:
        """Exports the reconstructed mesh to .ply, .obj, or .stl."""
        return o3d.io.write_triangle_mesh(filepath, mesh)
