from PIL import Image
import numpy as np
import cv2
import os
import logging
from typing import Optional
from pathlib import Path


def _add_text_to_image(img_np: np.ndarray, text: str) -> np.ndarray:
    """Adds standardized white text to the top-left corner of an image."""
    return cv2.putText(
        img_np.copy(),
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensures a numpy image array is 3-channel RGB."""
    if len(img_np.shape) == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np


def _create_heatmap(anom_map_norm_float: np.ndarray) -> np.ndarray:
    """Converts a 0-1 float anomaly map to an 8-bit JET colormap."""
    anom_map_u8 = (anom_map_norm_float * 255).astype(np.uint8)
    return cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)


def save_overlay_for_intro(
    path: str,
    img: Image.Image,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    kernel_size: int = 5,
    overlay_intensity: float = 0.4,
    display_scale: int = 1,
):
    """
    Saves a denoised, blended overlay for introductory figures.
    Assumes anom_map is a 0-1 normalized float array.
    """
    img_h, img_w = anom_map.shape
    img_w *= display_scale
    img_h *= display_scale
    img_np = np.array(img.resize((img_w, img_h), Image.Resampling.NEAREST))
    if display_scale != 1:
        anom_map = cv2.resize(
            anom_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
    img_np = _ensure_rgb(img_np)

    anom_map_u8 = (anom_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)
    try:
        _, binary_mask = cv2.threshold(
            anom_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except cv2.error:
        binary_mask = np.zeros_like(anom_map_u8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    denoised_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    denoised_mask = cv2.dilate(denoised_mask, kernel, iterations=1)
    overlay = cv2.addWeighted(
        img_np, (1.0 - overlay_intensity), heatmap, overlay_intensity, 0
    )
    mask_3d = _ensure_rgb(denoised_mask)
    final_image = np.where(mask_3d > 0, overlay, img_np)
    vis_dir = Path(outdir) / "intro_overlays" / category
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique filename like "contamination_001.png"
    p = Path(path)
    unique_filename = f"{p.parent.name}_{p.name}"

    out_path = vis_dir / unique_filename
    Image.fromarray(final_image).save(out_path)


def save_visualization(
    path: str,
    img: Image.Image,
    gt_mask: np.ndarray,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    vis_idx: int,
    saliency_mask: Optional[np.ndarray] = None,
    display_scale: int = 1,
):
    """Saves a 2x2 multi-panel visualization (Original, GT, Map, Saliency/Overlay)."""
    if display_scale < 1:
        raise ValueError("display_scale must be >= 1")
    target_shape = (
        anom_map.shape[1] * display_scale,
        anom_map.shape[0] * display_scale,
    )  # (W, H)
    target_shape_hw = (target_shape[1], target_shape[0])  # (H, W)

    img_np = np.array(img.resize(target_shape, Image.Resampling.NEAREST))
    img_np_rgb = _ensure_rgb(img_np)

    if display_scale != 1:
        anom_map = cv2.resize(
            anom_map, target_shape, interpolation=cv2.INTER_LINEAR
        )
    heatmap = _create_heatmap(anom_map)

    if gt_mask.shape != target_shape_hw:
        logging.warning(
            f"GT shape {gt_mask.shape} != Anom map shape {target_shape_hw}. Resizing GT."
        )
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            target_shape,
            interpolation=cv2.INTER_NEAREST,
        )
    gt_mask_vis = _ensure_rgb((gt_mask * 255).astype(np.uint8))
    panel1 = _add_text_to_image(img_np_rgb, "Original")
    panel2 = _add_text_to_image(gt_mask_vis, "Ground Truth")
    panel3 = _add_text_to_image(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        if saliency_mask.shape != target_shape_hw:
            saliency_mask = cv2.resize(
                saliency_mask, target_shape, interpolation=cv2.INTER_NEAREST
            )
        saliency_mask_u8 = (saliency_mask * 255).astype(np.uint8)
        saliency_mask_vis = _ensure_rgb(saliency_mask_u8)
        panel4 = _add_text_to_image(saliency_mask_vis, "Saliency Mask (FG)")
    else:
        overlay = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        panel4 = _add_text_to_image(overlay, "Overlay")
    combined_img = np.vstack([np.hstack([panel1, panel2]), np.hstack([panel3, panel4])])

    vis_dir = Path(outdir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{category}_example_{vis_idx}.png"
    Image.fromarray(combined_img).save(out_path)
