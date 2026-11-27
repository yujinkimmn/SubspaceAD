"""Independent image/pixel calibration; preserve PaDiM's model-resolution pred_score."""
import numpy as np
from sklearn.metrics import precision_recall_curve

NORMALIZED_THRESHOLD = .5
PADIM_IMAGE_SCORE_SOURCE = "model_input_map_max"
NATIVE_IMAGE_SCORE_SOURCE = "native_map_max"


def get_raw_image_scores(maps, raw_image_scores=None, *, calibration=None):
    """Prefer the model's explicit pred_score, never recompute it from a resized map."""
    if raw_image_scores is None:
        if calibration is not None and calibration.get("image_score_source") == PADIM_IMAGE_SCORE_SOURCE:
            raise ValueError("PaDiM image calibration requires model-input-resolution pred_score")
        return np.asarray(maps).max(axis=(1, 2))
    scores = np.asarray(raw_image_scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(maps) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite raw image score per anomaly map")
    return scores


def f1_adaptive_threshold(scores, labels):
    """Validation-only F1 maximum on raw scores, following anomalib's PR-curve rule."""
    precision, recall, thresholds = precision_recall_curve(labels.ravel(), scores.ravel())
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r)>0)
    return float(thresholds[np.argmax(f1)])


def calibrate_post_processor(rows, maps, *, raw_image_scores=None):
    """Fit independent image and native-pixel thresholds/min/max on validation only."""
    scores = get_raw_image_scores(maps, raw_image_scores)
    labels = np.asarray([r["label"] for r in rows])
    masks = np.stack([r["y"] > 0 for r in rows])
    return dict(source="validation", method="f1_adaptive", threshold=NORMALIZED_THRESHOLD,
                image_score_source=(PADIM_IMAGE_SCORE_SOURCE if raw_image_scores is not None else NATIVE_IMAGE_SCORE_SOURCE),
                image_threshold=f1_adaptive_threshold(scores, labels),
                image_min=float(scores.min()), image_max=float(scores.max()),
                pixel_threshold=f1_adaptive_threshold(maps, masks),
                pixel_min=float(maps.min()), pixel_max=float(maps.max()))


def normalize_scores(values, threshold, minimum, maximum):
    """Anomalib threshold-centered min/max normalization with clipping to [0, 1]."""
    values = np.asarray(values, dtype=np.float32)
    if maximum == minimum:
        # Constant calibration scores have no range; keep equality at 0.5.
        return np.where(values > threshold, 1., np.where(values < threshold, 0., .5)).astype(np.float32)
    return np.clip((values-threshold)/(maximum-minimum)+.5, 0., 1.)


def post_process(maps, calibration, *, raw_image_scores=None):
    raw_scores = get_raw_image_scores(maps, raw_image_scores, calibration=calibration)
    scores = normalize_scores(raw_scores, calibration["image_threshold"], calibration["image_min"], calibration["image_max"])
    normalized = normalize_scores(maps, calibration["pixel_threshold"], calibration["pixel_min"], calibration["pixel_max"])
    return scores, normalized, scores > NORMALIZED_THRESHOLD, normalized > NORMALIZED_THRESHOLD
