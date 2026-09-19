"""
Camera Calibration Module
Uses OpenCV to detect chessboard corners and compute camera intrinsics (K) and distortion coefficients (D).
Outputs parameters directly to YAML format compatible with the pipeline.

Scale Ambiguity Note:
Camera calibration determines optical ray angles (focal length, principal point) and lens distortion.
It does NOT impart metric scale to monocular video frames or depth maps.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml


def calibrate_from_images(
    image_paths: List[str],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 25.0,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Calibrate camera from a list of chessboard image paths.

    Args:
        image_paths: List of file paths to chessboard photos.
        pattern_size: Inner corners (columns, rows), e.g. (9, 6).
        square_size: Physical size of square side in user units (e.g., mm).
        visualize: If True, briefly displays detected corner overlays.

    Returns:
        camera_matrix (3x3 ndarray),
        dist_coeff (1x5 ndarray),
        reprojection_error (float),
        image_shape (width, height)
    """
    # Prepare 3D object points in board coordinate system (z = 0)
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"[Calibration] Processing {len(image_paths)} images with pattern {pattern_size}...")
    valid_count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [Warning] Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = (gray.shape[1], gray.shape[0])  # width, height

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(refined_corners)
            valid_count += 1

            if visualize:
                vis = cv2.drawChessboardCorners(img.copy(), pattern_size, refined_corners, ret)
                cv2.imshow("Chessboard Corners", vis)
                cv2.waitKey(200)

    if visualize:
        cv2.destroyAllWindows()

    if valid_count < 3:
        raise ValueError(
            f"Need at least 3 valid calibration frames with detected corners, found only {valid_count}. "
            "Ensure the chessboard is fully visible, well-lit, and matches the inner corner count."
        )

    print(f"[Calibration] Found corners in {valid_count}/{len(image_paths)} images. Computing calibration...")
    ret, camera_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    print(f"[Calibration] Calibration finished. RMS Reprojection Error: {ret:.4f} pixels")
    return camera_matrix, dist_coeff, ret, img_shape


def calibrate_from_video(
    video_path: str,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 25.0,
    frame_stride: int = 15,
    max_frames: int = 40,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Calibrate camera from a video of a chessboard by sampling frames periodically.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    temp_dir = "data/calibration/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    temp_paths = []
    for idx, f in enumerate(frames):
        p = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(p, f)
        temp_paths.append(p)

    try:
        res = calibrate_from_images(temp_paths, pattern_size, square_size, visualize)
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
    return res


def save_calibration(
    output_path: str,
    camera_matrix: np.ndarray,
    dist_coeff: np.ndarray,
    image_width: int,
    image_height: int,
    reprojection_error: Optional[float] = None,
    camera_name: str = "calibrated_camera",
) -> None:
    """Saves camera calibration parameters to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = {
        "camera_name": camera_name,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeff": dist_coeff.flatten().tolist(),
    }
    if reprojection_error is not None:
        data["rms_reprojection_error_px"] = float(reprojection_error)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None)
    print(f"[Calibration] Saved camera calibration parameters to: {output_path}")


def load_calibration(config_path: str) -> Dict:
    """
    Loads camera calibration parameters from a YAML file.
    Returns dictionary with 'camera_matrix' (3x3 float ndarray), 'dist_coeff' (1x5 float ndarray),
    'image_width', and 'image_height'.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Calibration file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeff = np.array(data["dist_coeff"], dtype=np.float64).reshape(-1)
    width = int(data.get("image_width", 1280))
    height = int(data.get("image_height", 720))

    return {
        "camera_matrix": camera_matrix,
        "dist_coeff": dist_coeff,
        "image_width": width,
        "image_height": height,
        "camera_name": data.get("camera_name", "camera"),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics using a chessboard pattern.")
    parser.add_argument("--images", type=str, help="Glob pattern for calibration images (e.g. 'data/calib/*.jpg')")
    parser.add_argument("--video", type=str, help="Path to calibration video file")
    parser.add_argument("--pattern", type=str, default="9x6", help="Inner chessboard corners (Cols x Rows), default: 9x6")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square dimension in mm or chosen units (default: 25.0)")
    parser.add_argument("--output", type=str, default="configs/camera.yaml", help="Output YAML file path")
    parser.add_argument("--vis", action="store_true", help="Display corner detections during calibration")
    args = parser.parse_args()

    cols, rows = map(int, args.pattern.split("x"))
    pattern_size = (cols, rows)

    if args.images:
        paths = glob.glob(args.images)
        if not paths:
            print(f"Error: No images found matching {args.images}")
            return
        K, D, rms, (w, h) = calibrate_from_images(paths, pattern_size, args.square_size, args.vis)
    elif args.video:
        K, D, rms, (w, h) = calibrate_from_video(args.video, pattern_size, args.square_size, visualize=args.vis)
    else:
        print("Please provide either --images or --video. Example:")
        print("  python src/calibration/calibrate_camera.py --images 'data/calibration/*.jpg' --pattern 9x6")
        return

    save_calibration(args.output, K, D, w, h, rms)
    print("\nCalibration successful!")
    print(f"Camera Matrix K:\n{K}")
    print(f"Distortion Coefficients:\n{D}")


if __name__ == "__main__":
    main()
