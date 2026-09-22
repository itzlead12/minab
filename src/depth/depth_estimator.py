"""
Monocular Relative Depth Estimation Module
Wraps Depth Anything V2 via Hugging Face Transformers.

CRITICAL RELATIVE DEPTH NOTICE:
Depth Anything V2 produces relative monocular depth, NOT metric measurements.
The model does not know physical scale (e.g. millimeters or meters).
Depth values are normalized and mapped to a relative working range [z_min, z_max] in arbitrary units.
Do NOT treat output depths as true physical distances.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "auto",
        relative_depth_min: float = 0.5,
        relative_depth_max: float = 5.0,
        max_inference_dimension: int = 518,
    ):
        """
        Initializes the Depth Anything V2 monocular relative depth estimator.

        Args:
            model_id: Hugging Face model identifier (e.g. 'depth-anything/Depth-Anything-V2-Small-hf').
            device: 'cuda', 'cpu', or 'auto' (selects GPU if available).
            relative_depth_min: Scaled relative near plane (arbitrary units).
            relative_depth_max: Scaled relative far plane (arbitrary units).
            max_inference_dimension: Maximum dimension for model inference downscaling.
        """
        self.model_id = model_id
        self.relative_depth_min = relative_depth_min
        self.relative_depth_max = relative_depth_max
        self.max_inference_dimension = max_inference_dimension

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DepthEstimator] Loading '{model_id}' on {self.device.upper()}...")
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("[DepthEstimator] Model loaded successfully.")

    def refine_depth_edges(
        self,
        norm_depth: np.ndarray,
        guide_image: np.ndarray,
        radius: int = 6,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Sharpens depth boundaries using guided filtering with the RGB frame as guidance.
        Aligns depth discontinuities to true object edges, eliminating depth halos.
        """
        if len(guide_image.shape) == 3:
            guide_gray = cv2.cvtColor(guide_image, cv2.COLOR_RGB2GRAY)
        else:
            guide_gray = guide_image

        if guide_gray.shape[:2] != norm_depth.shape[:2]:
            guide_gray = cv2.resize(guide_gray, (norm_depth.shape[1], norm_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        try:
            filtered = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=norm_depth.astype(np.float32),
                radius=radius,
                eps=eps,
            )
            return np.clip(filtered, 0.0, 1.0)
        except Exception:
            norm_u8 = (norm_depth * 255.0).astype(np.uint8)
            bilat = cv2.bilateralFilter(norm_u8, d=7, sigmaColor=50, sigmaSpace=7)
            return bilat.astype(np.float32) / 255.0

    def estimate(
        self,
        image: Union[np.ndarray, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        sharpen_edges: bool = True,
    ) -> np.ndarray:
        """
        Estimates a 2D relative depth map from an input RGB image with edge sharpening.

        Args:
            image: BGR/RGB numpy array (HxWx3) or PIL Image.
            target_size: Optional (width, height) to resize output depth map. If None, matches input image size.
            sharpen_edges: If True, applies edge-guided depth sharpening using RGB guidance.

        Returns:
            2D float32 numpy array (HxW) with scaled relative depth values in [relative_depth_min, relative_depth_max].
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert OpenCV BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            pil_image = Image.fromarray(rgb_image)
            orig_w, orig_h = image.shape[1], image.shape[0]
        else:
            pil_image = image
            rgb_image = np.array(pil_image)
            orig_w, orig_h = pil_image.size

        # Preprocess image
        inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate depth back to target or original size
        out_w, out_h = target_size if target_size is not None else (orig_w, orig_h)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = prediction.cpu().numpy().astype(np.float32)

        # Normalize relative depth to [0, 1]
        d_min = np.percentile(depth_np, 1.0)
        d_max = np.percentile(depth_np, 99.0)
        if d_max - d_min > 1e-6:
            norm_depth = np.clip((depth_np - d_min) / (d_max - d_min), 0.0, 1.0)
        else:
            norm_depth = np.zeros_like(depth_np)

        # Edge-guided sharpening using RGB guidance
        if sharpen_edges:
            norm_depth = self.refine_depth_edges(norm_depth, rgb_image)

        # Note: In Depth-Anything-V2, higher raw values correspond to nearer surfaces (disparity-like).
        # We invert so that 0 is near and 1 is far, then scale to [relative_depth_min, relative_depth_max].
        relative_z = self.relative_depth_min + (1.0 - norm_depth) * (self.relative_depth_max - self.relative_depth_min)
        return relative_z.astype(np.float32)

    @staticmethod
    def colorize_depth(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Colorizes a relative depth map into a 3-channel BGR visualization image.
        """
        d_min = np.min(depth_map)
        d_max = np.max(depth_map)
        if d_max - d_min > 1e-6:
            norm = ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(depth_map, dtype=np.uint8)

        colorized = cv2.applyColorMap(norm, colormap)
        return colorized
