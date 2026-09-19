"""Unit tests for feature tracker and relative monocular pose estimator."""

import numpy as np
import cv2
import pytest
from src.tracking.feature_tracker import FeatureTracker
from src.tracking.pose_estimator import PoseEstimator


def create_synthetic_pattern_image(shift_x=0, shift_y=0):
    """Creates a high-contrast textured test image with circles and lines."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = 50
    # Draw high-contrast grid and geometric shapes
    for x in range(50, 600, 40):
        for y in range(50, 440, 40):
            center = (x + shift_x, y + shift_y)
            cv2.circle(img, center, 8, (255, 255, 255), -1)
            cv2.circle(img, center, 4, (0, 0, 0), -1)
    cv2.rectangle(img, (100 + shift_x, 100 + shift_y), (300 + shift_x, 300 + shift_y), (200, 200, 200), 3)
    return img


def test_orb_extraction_and_matching():
    tracker = FeatureTracker(max_features=1000, match_ratio=0.8, min_matches=10)
    img1 = create_synthetic_pattern_image(shift_x=0, shift_y=0)
    img2 = create_synthetic_pattern_image(shift_x=10, shift_y=5)

    kp1, des1 = tracker.extract(img1)
    kp2, des2 = tracker.extract(img2)

    assert len(kp1) > 20, "Should detect sufficient ORB keypoints"
    assert des1 is not None and len(des1) == len(kp1)

    match_res = tracker.match(kp1, des1, kp2, des2)
    assert match_res is not None, "Matching should succeed between shifted patterns"
    assert len(match_res.matches) >= 10, "Should find at least 10 good correspondences"
    assert match_res.pts1.shape[1] == 2
    assert match_res.pts2.shape[1] == 2


def test_pose_estimator_scale_ambiguity_and_unit_norm():
    # Synthetic camera intrinsics
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    estimator = PoseEstimator(camera_matrix=K, min_inliers=8)

    # Generate synthetic 3D points in front of camera
    np.random.seed(42)
    pts_3d = np.random.uniform(-1.0, 1.0, (100, 3))
    pts_3d[:, 2] += 3.0 # Z in [2.0, 4.0]

    # Camera 1 (origin)
    p1_2d = (K @ pts_3d.T).T
    p1_2d = p1_2d[:, :2] / p1_2d[:, 2:]

    # Camera 2: rotate by 2 deg and translate in X by arbitrary amount (e.g. 0.8)
    angle = np.radians(2.0)
    R_true = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t_true = np.array([[0.8], [0.0], [0.0]]) # arbitrary metric translation

    pts_cam2 = (R_true @ pts_3d.T + t_true).T
    p2_2d = (K @ pts_cam2.T).T
    p2_2d = p2_2d[:, :2] / p2_2d[:, 2:]

    # Estimate motion from 2D correspondences
    pose = estimator.update_pose(frame_idx=1, pts1=p1_2d.astype(np.float32), pts2=p2_2d.astype(np.float32))

    assert pose is not None, "Pose estimation should succeed on clean correspondences"
    # Verify that monocular translation vector norm is exactly 1.0 (scale ambiguity)
    assert np.isclose(np.linalg.norm(pose.t_rel), 1.0, atol=1e-5), \
        "Monocular recoverPose must produce unit-norm translation vector (scale ambiguity)."
    # Verify trajectory tracking
    assert len(estimator.trajectory) == 2
