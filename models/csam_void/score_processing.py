"""Independent image/pixel calibration; preserve PaDiM's model-resolution pred_score."""
import math

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve

NORMALIZED_THRESHOLD = .5
PADIM_IMAGE_SCORE_SOURCE = "model_input_map_max"
NATIVE_IMAGE_SCORE_SOURCE = "native_map_max"
SMALL_REGIONS = ("1px", "2-4px")


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


def region_peaks(mask, scores):
    """Return (area, highest raw score) for each 8-connected ground-truth region."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count:
        ids = np.arange(1, count + 1)
        yield from zip(np.bincount(labels.ravel())[1:], ndimage.maximum(scores, labels, ids))


def threshold_at_recall(positive_scores, recall):
    """Highest inclusive threshold that attains recall on the supplied positives."""
    scores = np.asarray(positive_scores)
    total = len(scores)
    required = math.ceil(recall * total)
    if required > 1 and (required - 1) / total >= recall:
        required -= 1
    index = total - required
    return float(np.partition(scores, index)[index])


def calibrate_post_processor(rows, maps, *, raw_image_scores=None, goals=None):
    """Fit independent validation thresholds and ranges used to display scores."""
    scores = get_raw_image_scores(maps, raw_image_scores)
    labels = np.asarray([r["label"] for r in rows])
    masks = np.stack([r["y"] > 0 for r in rows])
    calibration = dict(source="validation", method="f1_adaptive", decision_rule="raw_gte",
                       threshold=NORMALIZED_THRESHOLD,
                       image_score_source=(PADIM_IMAGE_SCORE_SOURCE if raw_image_scores is not None else NATIVE_IMAGE_SCORE_SOURCE),
                       image_min=float(scores.min()), image_max=float(scores.max()),
                       pixel_min=float(maps.min()), pixel_max=float(maps.max()))
    if goals is None:
        calibration.update(image_threshold=f1_adaptive_threshold(scores, labels),
                           pixel_threshold=f1_adaptive_threshold(maps, masks))
        return calibration

    calibration.update(method="recall_constrained", goals=dict(goals),
                       image_threshold=threshold_at_recall(scores[labels == 1], goals["image_recall"]))
    peaks = {key: [] for key in SMALL_REGIONS}
    if goals["region_recall"] > 0:
        for mask, anomaly_map in zip(masks, maps):
            for area, peak in region_peaks(mask, anomaly_map):
                if area <= 4:
                    peaks["1px" if area == 1 else "2-4px"].append(peak)
    limits = [threshold_at_recall(values, goals["region_recall"])
              for values in peaks.values() if values]
    if limits:
        # Any higher threshold would miss at least one recall target. If the cap
        # fails here, there is no threshold attaining both recall and the cap.
        threshold = min(limits)
    else:
        threshold = f1_adaptive_threshold(maps, masks)
        background = maps[~masks]
        cap = goals["pixel_fpr_cap"]
        total = len(background)
        allowed = math.floor(cap * total)
        if allowed < total and (allowed + 1) / total <= cap:
            allowed += 1
        if allowed < len(background):
            index = len(background) - allowed - 1
            boundary = np.partition(background, index)[index]
            threshold = max(threshold, float(np.nextafter(boundary, background.dtype.type(np.inf))))
    calibration["pixel_threshold"] = threshold
    return calibration


def validation_constraints(metrics, goals):
    """True/False per goal, with None for region targets lacking ground truth."""
    checks = {"image_recall": metrics["recall"] >= goals["image_recall"],
              "pixel_fpr": metrics["pixel_fpr"] <= goals["pixel_fpr_cap"]}
    if goals["region_recall"] > 0:
        for key in SMALL_REGIONS:
            recall = metrics["regions"][key]["recall"]
            checks[key] = None if recall is None else recall >= goals["region_recall"]
    return checks


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
    if calibration.get("decision_rule") == "raw_gte":
        return scores, normalized, raw_scores >= calibration["image_threshold"], np.asarray(maps) >= calibration["pixel_threshold"]
    # Older checkpoints only stored the original normalized decision rule.
    return scores, normalized, scores > NORMALIZED_THRESHOLD, normalized > NORMALIZED_THRESHOLD
