"""Lightweight geometry tests; no model download required."""

import numpy as np
from PIL import Image

from src.subspacead.data.letterbox import (
    letterbox_geometry,
    letterbox_image,
    unletterbox_map,
    valid_patch_mask,
)
from src.subspacead.utils.viz import save_overlay_for_intro, save_visualization


def test_61_by_81_uses_side_padding_and_restores_original_shape():
    image = Image.new("RGB", (61, 81), (200, 100, 50))
    geometry = letterbox_geometry(image.size, 224)
    assert (geometry.width, geometry.height) == (169, 224)
    assert (geometry.left, geometry.top) == (27, 0)
    padded = np.asarray(letterbox_image(image, 224))
    assert padded.shape == (224, 224, 3)
    np.testing.assert_array_equal(padded[:, : geometry.left], 0)
    assert np.all(
        padded[:, geometry.left : geometry.left + 169] == [200, 100, 50]
    )

    map_on_canvas = np.zeros((224, 224), dtype=np.float32)
    map_on_canvas[:, geometry.left : geometry.left + geometry.width] = 2.0
    restored = unletterbox_map(map_on_canvas, image.size, 224)
    assert restored.shape == (81, 61)
    np.testing.assert_allclose(restored, 2.0)


def test_landscape_and_square_inputs():
    landscape = (81, 61)
    geometry = letterbox_geometry(landscape, 224)
    assert (geometry.width, geometry.height) == (224, 169)
    assert (geometry.left, geometry.top) == (0, 27)
    valid = valid_patch_mask(landscape, 224, 14)
    assert valid.shape == (16, 16)
    assert valid.sum() == 192
    assert valid_patch_mask((224, 224), 224, 14).all()
    assert unletterbox_map(np.ones((224, 224), dtype=np.float32), landscape, 224).shape == (61, 81)


def test_padded_token_columns_excluded():
    valid = valid_patch_mask((61, 81), 224, 14)
    assert valid.shape == (16, 16)
    assert not valid[:, :2].any()
    assert not valid[:, -2:].any()
    assert valid[:, 2:14].all()


def test_saved_visualizations_keep_rectangular_aspect_ratio(tmp_path):
    image = Image.new("RGB", (61, 81), (200, 100, 50))
    anomaly_map = np.linspace(0, 1, 61 * 81, dtype=np.float32).reshape(81, 61)
    gt_mask = np.zeros((81, 61), dtype=np.uint8)
    save_visualization(
        "sample.png", image, gt_mask, anomaly_map, str(tmp_path),
        "defect", 0, display_scale=4,
    )
    saved = Image.open(tmp_path / "visualizations" / "defect_example_0.png")
    assert saved.size == (2 * 61 * 4, 2 * 81 * 4)

    save_overlay_for_intro(
        "defect/sample.png", image, anomaly_map, str(tmp_path), "defect",
        display_scale=4,
    )
    saved_overlay = Image.open(
        tmp_path / "intro_overlays" / "defect" / "defect_sample.png"
    )
    assert saved_overlay.size == (61 * 4, 81 * 4)
