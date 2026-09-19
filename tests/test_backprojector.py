"""Unit tests for pinhole 3D backprojection."""

import numpy as np
import pytest
from src.reconstruction.backprojector import Backprojector


def test_backprojector_geometry():
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    backprojector = Backprojector(camera_matrix=K, pixel_stride=1, relative_depth_trunc=10.0, min_depth=0.1)

    h, w = 480, 640
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 2.5 # uniform relative depth = 2.5

    # Center pixel (cx, cy)
    rgb[int(cy), int(cx)] = [255, 0, 0] # Red

    pcd = backprojector.backproject(rgb, depth)
    pts = np.asarray(pcd.points)

    assert len(pts) == h * w, "All valid pixels should be backprojected"

    # Find the point corresponding to (cx, cy): should have X ~= 0, Y ~= 0, Z == 2.5
    center_pt = pts[int(cy) * w + int(cx)]
    assert np.isclose(center_pt[0], 0.0, atol=1e-4)
    assert np.isclose(center_pt[1], 0.0, atol=1e-4)
    assert np.isclose(center_pt[2], 2.5, atol=1e-4)


def test_backprojector_depth_filtering():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    backprojector = Backprojector(camera_matrix=K, pixel_stride=2, relative_depth_trunc=3.0, min_depth=1.0)

    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Set half depth values to 0.05 (too close) and half to 5.0 (beyond truncation)
    depth = np.ones((h, w), dtype=np.float32) * 0.05
    depth[:50, :50] = 2.0 # valid depth

    pcd = backprojector.backproject(rgb, depth)
    pts = np.asarray(pcd.points)

    # Only 50x50 with stride 2 = 25x25 = 625 points should pass
    assert len(pts) == 625
    assert np.all(pts[:, 2] == 2.0)
