"""
Feature Tracker Module
Extracts ORB features from consecutive video frames and matches them using BFMatcher with Hamming distance
and Lowe's ratio test for robust correspondence estimation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class MatchResult:
    """Holds correspondence results between two frames."""
    pts1: np.ndarray          # Nx2 float32 array of coordinates in frame 1
    pts2: np.ndarray          # Nx2 float32 array of coordinates in frame 2
    kp1: List[cv2.KeyPoint]   # Keypoints from frame 1
    kp2: List[cv2.KeyPoint]   # Keypoints from frame 2
    matches: List[cv2.DMatch] # Good DMatch objects passing the ratio test


class FeatureTracker:
    def __init__(
        self,
        max_features: int = 2000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        match_ratio: float = 0.75,
        min_matches: int = 25,
    ):
        """
        Initializes the ORB feature tracker.

        Args:
            max_features: Maximum number of features to retain.
            scale_factor: Pyramid decimation ratio (> 1.0).
            n_levels: Number of pyramid levels.
            match_ratio: Lowe's ratio test threshold.
            min_matches: Minimum number of verified matches required for valid tracking.
        """
        self.max_features = max_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.match_ratio = match_ratio
        self.min_matches = min_matches

        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances frame sharpness and micro-textures (CLAHE + unsharp masking)
        to maximize corner detection and feature descriptor distinctiveness.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 1. CLAHE to bring out subtle textures in shadows and highlights
        enhanced = self.clahe.apply(gray)

        # 2. Unsharp masking to sharpen edge gradients
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)
        return sharpened

    def extract(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detects ORB keypoints and computes their binary descriptors on sharpened frame.

        Args:
            frame: BGR or grayscale image array.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        sharpened_gray = self.preprocess_frame(frame)
        kp, des = self.orb.detectAndCompute(sharpened_gray, None)
        return kp, des

    def match(
        self,
        kp1: List[cv2.KeyPoint],
        des1: Optional[np.ndarray],
        kp2: List[cv2.KeyPoint],
        des2: Optional[np.ndarray],
        frame1_gray: Optional[np.ndarray] = None,
        frame2_gray: Optional[np.ndarray] = None,
    ) -> Optional[MatchResult]:
        """
        Matches descriptors between frame 1 and frame 2 using k-NN with k=2 and Lowe's ratio test.
        Optionally refines matched point coordinates with sub-pixel precision.

        Returns:
            MatchResult if number of valid matches >= min_matches, else None.
        """
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return None

        if len(des1) < 2 or len(des2) < 2:
            return None

        # k-NN match with k=2 for ratio test
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        pts1 = []
        pts2 = []

        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)

        if len(good_matches) < self.min_matches:
            return None

        pts1_arr = np.array(pts1, dtype=np.float32)
        pts2_arr = np.array(pts2, dtype=np.float32)

        # Sub-pixel corner refinement if grayscale frames provided
        if frame1_gray is not None and frame2_gray is not None and len(pts1_arr) >= 8:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            try:
                pts1_arr = cv2.cornerSubPix(frame1_gray, pts1_arr, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
                pts2_arr = cv2.cornerSubPix(frame2_gray, pts2_arr, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
            except Exception:
                pass

        return MatchResult(
            pts1=pts1_arr,
            pts2=pts2_arr,
            kp1=kp1,
            kp2=kp2,
            matches=good_matches,
        )

    def draw_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        match_result: MatchResult,
        max_draw: int = 80,
    ) -> np.ndarray:
        """Draws visual match lines between two frames for debugging/monitoring."""
        draw_matches = match_result.matches[:max_draw]
        vis = cv2.drawMatches(
            img1,
            match_result.kp1,
            img2,
            match_result.kp2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis
