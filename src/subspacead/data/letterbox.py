"""Aspect-ratio preserving preprocessing and anomaly-map coordinate restoration."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxGeometry:
    left: int
    top: int
    width: int
    height: int
    target_size: int


def letterbox_geometry(image_size: tuple[int, int], target_size: int) -> LetterboxGeometry:
    """Calculate centered padding for an image_size expressed as (width, height)."""
    width, height = image_size
    if width <= 0 or height <= 0 or target_size <= 0:
        raise ValueError("Image dimensions and target_size must be positive")
    scale = min(target_size / width, target_size / height)
    resized_width = min(target_size, max(1, round(width * scale)))
    resized_height = min(target_size, max(1, round(height * scale)))
    return LetterboxGeometry(
        left=(target_size - resized_width) // 2,
        top=(target_size - resized_height) // 2,
        width=resized_width,
        height=resized_height,
        target_size=target_size,
    )


def letterbox_image(image: Image.Image, target_size: int, pad_value: int = 0) -> Image.Image:
    """Resize uniformly and center on a square RGB canvas."""
    if not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be between 0 and 255")
    geometry = letterbox_geometry(image.size, target_size)
    resized = image.convert("RGB").resize(
        (geometry.width, geometry.height), Image.Resampling.BICUBIC
    )
    canvas = Image.new("RGB", (target_size, target_size), (pad_value,) * 3)
    canvas.paste(resized, (geometry.left, geometry.top))
    return canvas


def valid_patch_mask(
    image_size: tuple[int, int], target_size: int, patch_size: int
) -> np.ndarray:
    """Select patch tokens whose centers are inside the actual image, not padding."""
    if patch_size <= 0 or target_size % patch_size:
        raise ValueError("target_size must be divisible by positive patch_size")
    geometry = letterbox_geometry(image_size, target_size)
    centers = (np.arange(target_size // patch_size) + 0.5) * patch_size
    valid_x = (centers >= geometry.left) & (
        centers < geometry.left + geometry.width
    )
    valid_y = (centers >= geometry.top) & (
        centers < geometry.top + geometry.height
    )
    mask = valid_y[:, None] & valid_x[None, :]
    if not mask.any():
        raise ValueError("No valid image patches after letterboxing")
    return mask


def unletterbox_map(
    anomaly_map: np.ndarray, image_size: tuple[int, int], target_size: int
) -> np.ndarray:
    """Discard padded scores and restore the original (height, width) map."""
    if anomaly_map.shape != (target_size, target_size):
        raise ValueError(
            f"Expected {(target_size, target_size)} anomaly map; got {anomaly_map.shape}"
        )
    geometry = letterbox_geometry(image_size, target_size)
    roi = anomaly_map[
        geometry.top : geometry.top + geometry.height,
        geometry.left : geometry.left + geometry.width,
    ]
    width, height = image_size
    return cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
