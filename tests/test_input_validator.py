"""Unit tests for VideoInputAssessor and camera matrix scaling according to instruction.md."""

import os
import cv2
import numpy as np
import pytest
from src.calibration.input_validator import VideoInputAssessor
from src.calibration.calibrate_camera import scale_camera_matrix


@pytest.fixture
def sample_video_path(tmp_path):
    """Creates a short 720p synthetic test video with clear motion and texture."""
    video_file = str(tmp_path / "test_video_720p.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (1280, 720))

    # Render 30 frames of moving textured circle on gray background
    for i in range(30):
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        # Centered moving object
        cx = 640 + int(i * 3)
        cy = 360
        cv2.circle(frame, (cx, cy), 80, (220, 150, 50), -1)
        # Texture lines
        for step in range(-70, 70, 15):
            cv2.line(frame, (cx + step, cy - 60), (cx + step, cy + 60), (20, 20, 20), 2)
            cv2.circle(frame, (cx + step, cy), 4, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return video_file


def test_video_input_assessor(sample_video_path):
    assessor = VideoInputAssessor()
    report = assessor.assess_video(sample_video_path, sample_frames_count=10)

    assert report["resolution"] == [1280, 720]
    assert report["fps"] == 30.0
    assert report["total_frames"] == 30
    assert report["sharpness_score"] > 20.0
    assert "resolution_quality" in report
    assert "motion_type" in report
    assert report["subject_centered"] is True


def test_scale_camera_matrix():
    K_720p = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0],
    ])

    # Scale to 1080p (1920x1080)
    K_1080p = scale_camera_matrix(K_720p, orig_w=1280, orig_h=720, new_w=1920, new_h=1080)

    assert np.isclose(K_1080p[0, 0], 1000.0 * (1920 / 1280))
    assert np.isclose(K_1080p[1, 1], 1000.0 * (1080 / 720))
    assert np.isclose(K_1080p[0, 2], 640.0 * (1920 / 1280))
    assert np.isclose(K_1080p[1, 2], 360.0 * (1080 / 720))
    assert K_1080p[2, 2] == 1.0
