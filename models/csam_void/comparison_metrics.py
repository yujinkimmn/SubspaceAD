"""Collect compact paired-comparison evidence while native test maps are in memory."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

SCHEMA_VERSION = 1


def sample_evidence(row, normalized_map, image_prediction, threshold=.5, *, prediction_mask=None):
    """Native-grid region hits and background counts, without storing score maps."""
    gt = np.asarray(row["y"]) > 0
    norm = np.asarray(normalized_map)
    if norm.shape != gt.shape or not np.isfinite(norm).all() or np.any((norm < 0) | (norm > 1)):
        raise ValueError("Comparison requires finite native anomaly maps aligned to GT")
    pred = norm > threshold if prediction_mask is None else np.asarray(prediction_mask, dtype=bool)
    labels, count = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count+1)
    overlap = np.bincount(labels[pred], minlength=count+1)
    regions = []
    for i, bbox in enumerate(ndimage.find_objects(labels), 1):
        y, x = bbox
        regions.append({"area": int(areas[i]), "overlap": int(overlap[i]),
                        "bbox": [int(y.start), int(x.start), int(y.stop), int(x.stop)]})
    bg = norm[~gt]
    # The histogram is after validation-based pixel normalization, not raw distances.
    hist = np.histogram(bg, bins=np.linspace(0., 1., 11))[0]
    fingerprint = hashlib.sha256()
    fingerprint.update(str(gt.shape).encode("ascii"))
    fingerprint.update(np.ascontiguousarray(row["x"], dtype=np.float32).tobytes())
    fingerprint.update(gt.tobytes())
    return dict(id=str(row["id"]), fingerprint=fingerprint.hexdigest(), label=int(row["label"]),
                equipment=str(row["equipment"]), lot=str(row["lot"]), wafer=str(row.get("wafer", "")),
                image_prediction=bool(image_prediction), regions=regions, bg_pixels=int(bg.size),
                bg_fp=int(np.count_nonzero(pred & ~gt)), bg_score_sum=float(bg.sum(dtype=np.float64)),
                bg_hist=[int(v) for v in hist])


def write_evidence(path, samples):
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "split": "test",
                               "samples": samples}, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)
