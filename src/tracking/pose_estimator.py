"""
Monocular Camera Pose Estimation Module
Estimates camera rotation (R) and translation direction (t) between consecutive frames using Epipolar Geometry
via the Essential Matrix and OpenCV recoverPose().

CRITICAL MONOCULAR SCALE DISCLAIMER:
Monocular vision suffers from fundamental scale ambiguity.
`cv2.recoverPose()` normalizes the translation vector to unit length (||t|| = 1.0).
Absolute metric distance cannot be recovered from single-camera video alone without external reference
(e.g., metric markers, stereo baseline, or calibrated IMU).
All camera poses and resulting 3D reconstructions are in ARBITRARY RELATIVE UNITS, not meters.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FramePose:
    """Represents an estimated camera pose in relative world coordinates."""
    frame_idx: int
    # 4x4 Transformation matrix T_world_camera: transforms a point from camera frame to world frame: P_w = T_w_c @ P_c
    transform_matrix: np.ndarray
    R_rel: np.ndarray        # 3x3 relative rotation from previous frame
    t_rel: np.ndarray        # 3x1 relative translation direction (||t|| = 1.0, scale ambiguous)
    inliers_count: int       # Number of RANSAC inliers supporting this pose
    is_keyframe: bool = False

    @property
    def camera_center(self) -> np.ndarray:
        """Camera optical center in world coordinates."""
        return self.transform_matrix[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """3x3 Camera orientation matrix in world coordinates."""
        return self.transform_matrix[:3, :3]


class PoseEstimator:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeff: Optional[np.ndarray] = None,
        ransac_prob: float = 0.999,
        ransac_threshold_px: float = 1.2,
        min_inliers: int = 20,
    ):
        """
        Initializes the monocular relative pose estimator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix K.
            dist_coeff: Distortion coefficients vector (or None if pre-rectified).
            ransac_prob: Desired probability that the RANSAC algorithm produces a useful result.
            ransac_threshold_px: Maximum pixel distance from an epipolar line to consider a point an inlier.
            min_inliers: Minimum number of RANSAC inliers to accept a pose estimation.
        """
        self.K = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeff = np.array(dist_coeff, dtype=np.float64) if dist_coeff is not None else None
        self.ransac_prob = ransac_prob
        self.ransac_threshold_px = ransac_threshold_px
        self.min_inliers = min_inliers

        # World-to-current camera pose accumulator.
        # Initial frame (frame 0) is world origin: T_w_c0 = Identity(4x4)
        self.current_T_world_camera = np.eye(4, dtype=np.float64)
        self.trajectory: List[FramePose] = [
            FramePose(
                frame_idx=0,
                transform_matrix=self.current_T_world_camera.copy(),
                R_rel=np.eye(3, dtype=np.float64),
                t_rel=np.zeros((3, 1), dtype=np.float64),
                inliers_count=0,
                is_keyframe=True,
            )
        ]

    def reset(self):
        """Resets the accumulated trajectory to the origin."""
        self.current_T_world_camera = np.eye(4, dtype=np.float64)
        self.trajectory = [
            FramePose(
                frame_idx=0,
                transform_matrix=self.current_T_world_camera.copy(),
                R_rel=np.eye(3, dtype=np.float64),
                t_rel=np.zeros((3, 1), dtype=np.float64),
                inliers_count=0,
                is_keyframe=True,
            )
        ]

    def estimate_relative_motion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
        """
        Estimates relative rotation R and translation direction t between two sets of matched points.

        Returns:
            Tuple of (R_rel, t_rel, inlier_count, inlier_mask) or None if estimation fails.
            NOTE: t_rel is normalized to unit length: ||t_rel|| == 1.0 (monocular scale ambiguity).
        """
        if len(pts1) < 8 or len(pts2) < 8:
            return None

        # Undistort 2D points if lens distortion coefficients are supplied
        if self.dist_coeff is not None and np.any(self.dist_coeff != 0):
            p1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, self.dist_coeff, P=self.K).reshape(-1, 2)
            p2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, self.dist_coeff, P=self.K).reshape(-1, 2)
        else:
            p1_undist = pts1
            p2_undist = pts2

        # 1. Compute Essential Matrix using 5-point algorithm with RANSAC
        E, mask_e = cv2.findEssentialMat(
            p1_undist,
            p2_undist,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold_px,
        )

        if E is None or E.shape != (3, 3):
            return None

        # 2. Decompose Essential Matrix into R and unit translation t via chirality check
        num_inliers, R, t, mask_pose = cv2.recoverPose(
            E,
            p1_undist,
            p2_undist,
            cameraMatrix=self.K,
            mask=mask_e,
        )

        if num_inliers < self.min_inliers:
            return None

        # Ensure t is a clean 3x1 unit vector
        t = t.reshape(3, 1)
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-7:
            t = t / norm_t

        return R, t, num_inliers, mask_pose

    def update_pose(
        self,
        frame_idx: int,
        pts1: np.ndarray,
        pts2: np.ndarray,
        is_keyframe: bool = False,
    ) -> Optional[FramePose]:
        """
        Updates the global camera pose by chaining relative motion from matched correspondences.

        Epipolar transformation convention:
            Points in frame k (current) are related to frame k-1 (previous) by:
                P_k = R @ P_{k-1} + t
            Therefore:
                P_{k-1} = R^T @ (P_k - t) = R^T @ P_k - R^T @ t
            In 4x4 matrix form:
                T_{k-1, k} = [R, t; 0, 1]
                T_{k, k-1} = inv(T_{k-1, k}) = [R^T, -R^T @ t; 0, 1]
            The camera-to-world pose accumulates as:
                T_w^k = T_w^{k-1} @ T_{k, k-1}
        """
        result = self.estimate_relative_motion(pts1, pts2)
        if result is None:
            return None

        R, t, inliers, _ = result

        # Invert relative transform to obtain step from current frame to previous frame in world chain
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R.T
        T_rel[:3, 3:] = -R.T @ t

        # Accumulate: T_w_current = T_w_previous @ T_rel
        self.current_T_world_camera = self.current_T_world_camera @ T_rel

        frame_pose = FramePose(
            frame_idx=frame_idx,
            transform_matrix=self.current_T_world_camera.copy(),
            R_rel=R,
            t_rel=t,
            inliers_count=inliers,
            is_keyframe=is_keyframe,
        )
        self.trajectory.append(frame_pose)
        return frame_pose

    def get_trajectory_points(self) -> np.ndarray:
        """Returns an Nx3 array of camera optical centers across all recorded poses."""
        return np.array([p.camera_center for p in self.trajectory], dtype=np.float64)
