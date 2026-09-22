import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class VideoInputAssessor:


    def __init__(
        self,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 30.0,
        recommended_min_duration: float = 5.0,
        recommended_max_duration: float = 15.0,
        min_sharpness_score: float = 80.0,
        min_parallax_displacement_px: float = 3.0,
        max_parallax_displacement_px: float = 80.0,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.recommended_min_duration = recommended_min_duration
        self.recommended_max_duration = recommended_max_duration
        self.min_sharpness_score = min_sharpness_score
        self.min_parallax_displacement_px = min_parallax_displacement_px
        self.max_parallax_displacement_px = max_parallax_displacement_px

    def assess_video(self, video_path: str, sample_frames_count: int = 30) -> Dict:

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        # Sample frames evenly for analysis
        step = max(1, total_frames // sample_frames_count)
        sampled_frames = []
        frame_indices = []

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and len(sampled_frames) < sample_frames_count:
                sampled_frames.append(frame)
                frame_indices.append(idx)
            idx += 1

        cap.release()

        # 1. Resolution Check
        is_hd = (width >= 1280 and height >= 720)
        is_1080p = (width >= 1920 and height >= 1080)
        res_label = f"{width}x{height}"
        if is_1080p:
            res_quality = "1080p (Excellent)"
        elif is_hd:
            res_quality = "720p (Good)"
        else:
            res_quality = f"Sub-HD ({res_label}, Recommended: 720p or 1080p)"

        # 2. Duration & FPS Checks
        warnings: List[str] = []
        passed_checks: List[str] = []
        tips: List[str] = []

        if 23.0 <= fps <= 31.0:
            passed_checks.append(f"Frame rate is optimal: {fps:.1f} fps (24-30 fps expected)")
        else:
            warnings.append(f"Frame rate is {fps:.1f} fps. Standard 24–30 fps provides optimal temporal spacing.")

        if self.recommended_min_duration <= duration <= self.recommended_max_duration:
            passed_checks.append(f"Duration is ideal: {duration:.1f}s (5-15s recommended)")
        elif duration < self.min_duration_sec:
            warnings.append(f"Duration is very short ({duration:.1f}s). System needs 5–15s for full scene coverage.")
        elif duration > self.max_duration_sec:
            warnings.append(f"Duration is long ({duration:.1f}s). Processing may take longer; consider trimming.")
        else:
            passed_checks.append(f"Duration acceptable: {duration:.1f}s")

        # 3. Sharpness Assessment (Laplacian Variance)
        sharpness_scores = []
        center_densities = []

        for f in sampled_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # Laplacian variance measures edge crispness (focus & blur)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_scores.append(lap_var)

            # Check keypoint density in center region (inner 50% width and height)
            orb = cv2.ORB_create(nfeatures=500)
            kps = orb.detect(gray, None)
            if kps:
                cx_min, cx_max = width * 0.25, width * 0.75
                cy_min, cy_max = height * 0.25, height * 0.75
                center_kps = [k for k in kps if cx_min <= k.pt[0] <= cx_max and cy_min <= k.pt[1] <= cy_max]
                center_densities.append(len(center_kps) / len(kps))

        avg_sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0
        avg_center_ratio = float(np.mean(center_densities)) if center_densities else 0.5

        if avg_sharpness >= 200.0:
            sharpness_grade = "Crisp & Sharp"
            passed_checks.append(f"High image sharpness score: {avg_sharpness:.1f} (stable focus)")
        elif avg_sharpness >= self.min_sharpness_score:
            sharpness_grade = "Acceptable Focus"
            passed_checks.append(f"Moderate sharpness score: {avg_sharpness:.1f}")
        else:
            sharpness_grade = "Soft / Blurry"
            warnings.append(f"Low sharpness score: {avg_sharpness:.1f}. Avoid motion blur or out-of-focus capture.")
            tips.append("Hold camera steadily or clean lens to prevent motion blur (instruction.md Section 4).")

        if avg_center_ratio >= 0.45:
            passed_checks.append(f"Subject well-centered: {avg_center_ratio * 100:.0f}% of features in center region")
            subject_centered = True
        else:
            subject_centered = False
            tips.append("Keep the main subject centered throughout the camera motion (instruction.md Section 3).")

        # 4. Parallax & Motion Quality (Frame-to-Frame Optical Flow)
        displacements = []
        if len(sampled_frames) >= 2:
            prev_gray = cv2.cvtColor(sampled_frames[0], cv2.COLOR_BGR2GRAY)
            for f in sampled_frames[1:]:
                curr_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                # Compute good features to track and sparse optical flow
                p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
                if p0 is not None and len(p0) > 10:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
                    if p1 is not None and st is not None:
                        good_prev = p0[st == 1]
                        good_curr = p1[st == 1]
                        if len(good_prev) > 5:
                            mag = np.linalg.norm(good_curr - good_prev, axis=1)
                            displacements.append(float(np.median(mag)))
                prev_gray = curr_gray

        avg_parallax = float(np.mean(displacements)) if displacements else 0.0

        if avg_parallax < self.min_parallax_displacement_px:
            motion_type = "Static / Insufficient Parallax"
            warnings.append(f"Camera motion is nearly static ({avg_parallax:.1f}px displacement). Parallax is required!")
            tips.append("Move the camera in a smooth arc around the object; avoid static shooting (instruction.md Section 1).")
        elif avg_parallax > self.max_parallax_displacement_px:
            motion_type = "Excessive Speed / Jerky"
            warnings.append(f"Camera motion is very fast ({avg_parallax:.1f}px displacement). May cause tracking loss.")
            tips.append("Move the camera slowly and smoothly (instruction.md Section 3).")
        else:
            motion_type = "Smooth Continuous Arc"
            passed_checks.append(f"Good motion parallax detected: {avg_parallax:.1f}px average displacement")

        # Overall recommendation
        is_compliant = len(warnings) <= 1 and avg_sharpness >= self.min_sharpness_score and avg_parallax >= self.min_parallax_displacement_px

        return {
            "video_path": video_path,
            "resolution": [width, height],
            "resolution_label": res_label,
            "resolution_quality": res_quality,
            "fps": round(fps, 2),
            "duration_sec": round(duration, 2),
            "total_frames": total_frames,
            "sharpness_score": round(avg_sharpness, 1),
            "sharpness_grade": sharpness_grade,
            "parallax_score": round(avg_parallax, 1),
            "motion_type": motion_type,
            "subject_centered": subject_centered,
            "center_feature_ratio": round(avg_center_ratio, 2),
            "is_compliant": is_compliant,
            "passed_checks": passed_checks,
            "warnings": warnings,
            "tips": tips,
        }
