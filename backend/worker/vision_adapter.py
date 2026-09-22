import os
import sys
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("minab.worker.vision_adapter")


def get_python_executable() -> str:
    """
    Finds the appropriate Python executable.
    Prefers project .venv Python if present to guarantee PyTorch/Torchvision dependencies.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_win = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    venv_posix = os.path.join(root_dir, ".venv", "bin", "python")

    if os.path.exists(venv_win):
        return venv_win
    elif os.path.exists(venv_posix):
        return venv_posix

    return sys.executable


def run_vision_pipeline(
    video_path: str,
    output_dir: str,
    camera_config: str = "configs/camera_default.yaml",
    frame_stride: int = 2,
    max_frames: Optional[int] = None,
    progress_json_path: Optional[str] = None,
) -> None:
    """
    Executes the monocular 3D reconstruction pipeline as a headless subprocess.
    """
    python_exe = get_python_executable()
    cmd = [
        python_exe,
        "src/pipeline.py",
        "--video", video_path,
        "--camera", camera_config,
        "--output-dir", output_dir,
        "--frame-stride", str(frame_stride),
        "--headless",
    ]

    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])

    if progress_json_path:
        cmd.extend(["--progress-json", progress_json_path])

    logger.info(f"Running vision command: {' '.join(cmd)}")

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        logger.error(f"Vision pipeline failed with return code {res.returncode}:\n{res.stderr}")
        raise RuntimeError(f"Vision pipeline execution failed: {res.stderr or res.stdout}")

    logger.info("Vision pipeline completed successfully.")
