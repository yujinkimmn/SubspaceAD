"""CSAM experiment runner. Local data only; never uploads images."""
import argparse
import csv
import hashlib
import html
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from visualization import jet_heatmap, panel_metadata_lines, save_panel
from csam_progress import LOG, setup_logging, console_info, epoch_note, stage_progress
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

H, W = 61, 81
PADIM_BACKBONE = "resnet18.a1_in1k"
NORMALIZED_THRESHOLD = .5
EXPS = ["exp0_padim", "exp1_cnn", "exp2_normals", "exp3_hard_negative",
        "exp4_copy_paste", "exp5_domain_aug", "exp6_mixstyle", "exp7_semi"]

def elapsed(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"

def pct(x):
    return "N/A" if x is None else f"{100*x:.2f}%"

def number(x):
    return "N/A" if x is None else f"{x:.5f}"

def gpu_memory(device):
    if device.type != "cuda":
        return "device=CPU"
    try:
        mib = 1024**2
        return (f"CUDA allocated={torch.cuda.memory_allocated(device)/mib:.0f} MiB "
                f"reserved={torch.cuda.memory_reserved(device)/mib:.0f} MiB "
                f"peak_allocated={torch.cuda.max_memory_allocated(device)/mib:.0f} MiB "
                f"total={torch.cuda.get_device_properties(device).total_memory/mib:.0f} MiB")
    except Exception:
        return "CUDA memory unavailable"

def log_dataset(parts):
    for split, rows in parts.items():
        groups = [("ALL", rows)] + [(eq, [r for r in rows if r["equipment"] == eq])
                                    for eq in sorted({r["equipment"] for r in rows})]
        for eq, rr in groups:
            n = sum(r["label"] == 0 for r in rr)
            v = sum(r["label"] == 1 for r in rr)
            u = sum(r["label"] == -1 for r in rr)
            LOG.info("[Data] %s equipment=%s total=%d normal=%d void=%d unlabeled=%d lots=%d",
                     split, eq, len(rr), n, v, u, len({r["lot"] for r in rr}))

def metric_line(m, th):
    return (f"recall={pct(m['recall'])} FPR={pct(m['fpr'])} "
            f"worst_FPR={pct(m['worst_equipment_fpr'])} Dice={number(m['dice'])} "
            f"IoU={number(m['iou'])} 1px={pct(m['regions']['1px']['recall'])} "
            f"2-4px={pct(m['regions']['2-4px']['recall'])} "
            f"5+px={pct(m['regions']['5+px']['recall'])} "
            f"FP={m['fp']} FN={m['fn']} threshold={th:.6g}")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_image(path, mask=False, expected_shape=None, return_raw=False):
    with Image.open(path) as im:
        a = np.asarray(im)
    allowed = (expected_shape,) if expected_shape is not None else ((H, W), (W, H))
    if a.shape not in allowed:
        raise ValueError(f"{path}: expected grayscale H,W in {allowed}, got {a.shape}")
    if mask:
        return (a > 0).astype(np.float32)
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{path}: only uint8/uint16 grayscale supported; got {a.dtype}")
    scaled = a.astype(np.float32) / np.iinfo(a.dtype).max
    return (scaled, a.copy()) if return_raw else scaled

def load_manifest(path):
    path = Path(path).resolve()
    rows, groups, hashes = [], {}, {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "mask", "label", "split", "equipment", "lot"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV columns: {sorted(required)}")
        for i, r in enumerate(reader, 2):
            split = r["split"].strip().lower()
            split = "train" if split == "ref" else split
            label = int(r["label"])
            eq, lot = r["equipment"].strip(), r["lot"].strip()
            # Optional display metadata, not a split/label key. Preserve leading zeros.
            wafer = (r.get("wafer") or "").strip() or (r.get("wafer_id") or "").strip()
            if split not in ("train", "val", "test") or label not in (-1, 0, 1) or not eq or not lot:
                raise ValueError(f"Bad manifest row {i}")
            if label == -1 and split != "train":
                raise ValueError("Unlabeled samples allowed only in train")
            p = (path.parent / r["image"]).resolve()
            x, raw = read_image(p, expected_shape=rows[0]["x"].shape if rows else None, return_raw=True)
            m = read_image((path.parent / r["mask"]).resolve(), True, expected_shape=x.shape) if r["mask"].strip() else np.zeros_like(x)
            if label == 1 and not m.any():
                raise ValueError(f"Positive requires a nonempty mask: row {i}")
            if label != 1 and m.any():
                raise ValueError(f"Normal/unlabeled row has positive mask: {i}")
            # Use globally unique ROOT lot IDs supplied by user, not equipment+lot.
            if lot in groups and groups[lot] != split:
                raise ValueError(f"Lot leakage: {lot}")
            groups[lot] = split
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image rows {hashes[digest]} and {i}; deduplicate before running")
            hashes[digest] = i
            rows.append(dict(id=str(i), image=str(p), x=x, raw=raw, y=m, label=label,
                             split=split, equipment=eq, lot=lot, wafer=wafer))
    if not rows:
        raise ValueError("Empty manifest")
    return rows

def split_rows(rows, heldout=None):
    out = {}
    for s in ("train", "val", "test"):
        out[s] = [r for r in rows if r["split"] == s and
                  (heldout is None or (r["equipment"] == heldout if s == "test" else r["equipment"] != heldout))]
        labels = {r["label"] for r in out[s]}
        if not {0, 1}.issubset(labels):
            raise ValueError(f"{s} requires confirmed normal and masked void samples (holdout={heldout})")
    return out

def domain_aug(x):
    # Photometric changes only; no resize, blur, or geometry that deletes 1px defects.
    gain = random.uniform(.85, 1.15)
    offset = random.uniform(-.06, .06)
    # Mild unsharp masking, without spatially shifting the mask.
    sharp = random.uniform(0, .25)
    smooth = ndimage.gaussian_filter(x, .65)
    return np.clip(gain * (x + sharp * (x - smooth)) + offset, 0, 1).astype(np.float32)

def copy_void(background, donor):
    # Residual transfer rather than copying source background pixels.
    x, m = background.copy(), donor["y"]
    labels, n = ndimage.label(m, structure=np.ones((3, 3)))
    ids = np.arange(1, n + 1)
    random.shuffle(ids)
    for k in ids:
        yy, xx = np.where(labels == k)
        if not len(yy):
            continue
        ring = ndimage.binary_dilation(labels == k, iterations=2) & (m == 0)
        if not ring.any():
            continue
        residual = donor["x"][yy, xx] - np.median(donor["x"][ring])
        # Reject components without a measurable positive bright residual.
        if not np.all(residual > 0):
            continue
        # Keep source coordinates: avoids assuming every chip region is physically valid.
        local = ndimage.binary_dilation(labels == k, iterations=2) & (labels != k)
        base = np.median(x[local]) if local.any() else np.median(x)
        values = np.clip(base + residual * random.uniform(.85, 1.15), 0, 1)
        if np.any(values <= x[yy, xx] + 1e-5):
            continue
        x[yy, xx] = values
        y = np.zeros_like(m)
        y[yy, xx] = 1
        return x, y
    return None

class Samples(Dataset):
    def __init__(self, rows, donors=(), paste=False, aug=False):
        self.rows, self.donors = rows, list(donors)
        self.paste, self.aug = paste, aug
        self.paste_attempts = self.paste_successes = self.aug_count = 0
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x, y = r["x"].copy(), r["y"].copy()
        if self.paste and r["label"] == 0 and self.donors and random.random() < .5:
            self.paste_attempts += 1
            pair = copy_void(x, random.choice(self.donors))
            if pair is not None:
                self.paste_successes += 1
                x, y = pair
        if self.aug:
            self.aug_count += 1
            x = domain_aug(x)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])

class MixStyle(nn.Module):
    def __init__(self, p=.5, alpha=.1):
        super().__init__()
        self.p, self.alpha = p, alpha
        self.applied = 0
    def forward(self, x):
        if not self.training or len(x) < 2 or random.random() >= self.p:
            return x
        self.applied += 1
        mu = x.mean((2, 3), keepdim=True).detach()
        sig = (x.var((2, 3), unbiased=False, keepdim=True) + 1e-6).sqrt().detach()
        perm = torch.randperm(len(x), device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((len(x), 1, 1, 1)).to(x.device)
        return (x - mu) / sig * (lam * sig + (1-lam) * sig[perm]) + lam * mu + (1-lam) * mu[perm]

class VoidNet(nn.Module):
    def __init__(self, mix=False):
        super().__init__()
        def conv(a, b, d=1):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=d, dilation=d), nn.ReLU())
        self.first = conv(1, 16)
        self.mix = MixStyle() if mix else nn.Identity()
        self.rest = nn.Sequential(conv(16,32), conv(32,32,2), conv(32,32,4),
                                  conv(32,16), nn.Conv2d(16,1,1))
    def forward(self, x):
        return self.rest(self.mix(self.first(x)))

def loss_fn(logits, mask, hard=False):
    # Per-class averaging avoids 1/4941 foreground pixels being drowned out.
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pos, neg = mask > .5, mask <= .5
    parts = []
    if pos.any():
        parts.append(bce[pos].mean())
    if neg.any():
        parts.append(bce[neg].mean())
        if hard:
            values = bce[neg]
            k = max(1, int(.01 * values.numel()))
            parts.append(values.topk(k).values.mean())
    prob = logits.sigmoid()
    # Dice on positive images only; empty masks handled by BCE.
    sel = mask.flatten(1).sum(1) > 0
    dice = logits.new_zeros(())
    if sel.any():
        p, y = prob[sel], mask[sel]
        dice = 1 - ((2*(p*y).flatten(1).sum(1)+1e-6) /
                    ((p+y).flatten(1).sum(1)+1e-6)).mean()
    return sum(parts) / len(parts) + dice

@torch.no_grad()
def cnn_maps(model, rows, args, device, desc=None, batch_done=None):
    model.eval()
    maps = []
    with stage_progress(desc,len(rows),unit="img") as bar:
        for x, _ in DataLoader(Samples(rows), batch_size=args.batch):
            maps.extend(model(x.to(device)).sigmoid().cpu().numpy()[:,0])
            bar.update(len(x))
            if batch_done is not None:
                batch_done()
    return np.asarray(maps)

def threshold_for(rows, maps, recall, region_recall=.90):
    positive = sorted((float(m.max()) for r,m in zip(rows,maps) if r["label"] == 1), reverse=True)
    if not positive:
        raise ValueError("Validation has no positive images")
    thresholds = [positive[max(0, math.ceil(recall * len(positive))-1)]]
    buckets = defaultdict(list)
    if region_recall > 0:
        for r,m in zip(rows,maps):
            lab,n = ndimage.label(r["y"] > 0, structure=np.ones((3,3)))
            for j in range(1,n+1):
                obj = lab == j
                area = obj.sum()
                if area <= 4:
                    buckets["1px" if area == 1 else "2-4px"].append(float(m[obj].max()))
        for vals in buckets.values():
            vals.sort(reverse=True)
            thresholds.append(vals[math.ceil(region_recall * len(vals))-1])
    return min(thresholds)

def f1_adaptive_threshold(scores, labels):
    """Validation-only F1 maximum on raw scores, following anomalib's PR-curve rule."""
    precision, recall, thresholds = precision_recall_curve(labels.ravel(), scores.ravel())
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r)>0)
    return float(thresholds[np.argmax(f1)])

def calibrate_post_processor(rows, maps):
    """Fit independent image and pixel thresholds/min/max on validation only."""
    scores = maps.max(axis=(1, 2))
    labels = np.asarray([r["label"] for r in rows])
    masks = np.stack([r["y"] > 0 for r in rows])
    return dict(source="validation", method="f1_adaptive", threshold=NORMALIZED_THRESHOLD,
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

def post_process(maps, calibration):
    raw_scores = maps.max(axis=(1, 2))
    scores = normalize_scores(raw_scores, calibration["image_threshold"], calibration["image_min"], calibration["image_max"])
    normalized = normalize_scores(maps, calibration["pixel_threshold"], calibration["pixel_min"], calibration["pixel_max"])
    return scores, normalized, scores > NORMALIZED_THRESHOLD, normalized > NORMALIZED_THRESHOLD

def pixel_metrics(rows, maps, include_aupro=False):
    """Dataset-level pixel AUROC and exact region-balanced partial ROC area."""
    target = np.stack([r["y"] > 0 for r in rows])
    scores = np.asarray(maps)
    if target.shape != scores.shape:
        raise ValueError("Pixel metric requires aligned native anomaly maps and masks")
    y = target.ravel()
    z = scores.ravel()
    result = {"pixel_auroc": float(roc_auc_score(y, z)) if y.any() and not y.all() else None}
    if not include_aupro:
        return result
    result["pixel_aupro"] = None
    if not y.any() or y.all():
        return result
    # Negative pixels have equal weight; each 8-connected positive region
    # has total weight 1, independently of its area. Weighted ROC TPR is PRO.
    weights = np.ones(target.shape, dtype=np.float64)
    for i, mask in enumerate(target):
        labels, _ = ndimage.label(mask, structure=np.ones((3,3)))
        counts = np.bincount(labels.ravel())
        weights[i][mask] = 1.0 / counts[labels[mask]]
    fpr, pro, _ = roc_curve(y, z, sample_weight=weights.ravel(), drop_intermediate=False)
    limit = .30
    # Include vertical segments and linearly interpolate at the integration limit.
    end = np.searchsorted(fpr, limit, side="right")
    xf, yf = fpr[:end], pro[:end]
    if xf[-1] < limit:
        xf = np.append(xf, limit)
        yf = np.append(yf, pro[end-1] + (pro[end]-pro[end-1]) *
                       (limit-fpr[end-1]) / (fpr[end]-fpr[end-1]))
    result["pixel_aupro"] = float(np.clip(np.sum(np.diff(xf)*(yf[1:]+yf[:-1])*.5)/limit, 0, 1))
    return result


def summarize(rows, maps, threshold, evaluation_split=None, *, image_scores=None, pred_labels=None, pred_masks=None, metric_maps=None):
    tp = fp = tn = fn = inter = union = predpix = truepix = 0
    region = {k:[0,0] for k in ("1px", "2-4px", "5+px")}
    false_regions = 0
    labels, scores = [], []
    for i,(r,m) in enumerate(zip(rows,maps)):
        pred = m >= threshold if pred_masks is None else pred_masks[i]
        actual = r["label"] == 1
        flag = bool(pred.any()) if pred_labels is None else bool(pred_labels[i])
        tp += int(actual and flag); fn += int(actual and not flag)
        fp += int(not actual and flag); tn += int(not actual and not flag)
        labels.append(int(actual)); scores.append(float(m.max() if image_scores is None else image_scores[i]))
        y = r["y"] > 0
        inter += int((pred & y).sum()); union += int((pred | y).sum())
        predpix += int(pred.sum()); truepix += int(y.sum())
        if not actual:
            false_regions += ndimage.label(pred, structure=np.ones((3,3)))[1]
        lab,n = ndimage.label(y, structure=np.ones((3,3)))
        for j in range(1,n+1):
            obj = lab == j
            area = obj.sum()
            bucket = "1px" if area == 1 else "2-4px" if area <= 4 else "5+px"
            region[bucket][1] += 1
            region[bucket][0] += int((obj & pred).any())
    div = lambda a,b: float(a/b) if b else None
    result = dict(tp=tp, fp=fp, tn=tn, fn=fn, recall=div(tp,tp+fn), fpr=div(fp,fp+tn),
                underkill=div(fn,tp+fn), dice=div(2*inter,predpix+truepix), iou=div(inter,union),
                auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                fp_regions_per_normal=div(false_regions,fp+tn),
                regions={k:dict(hit=v[0], count=v[1], recall=div(*v)) for k,v in region.items()})

    if evaluation_split is not None:
        result["image_auroc"] = result["auroc"]
        result.update(pixel_metrics(rows, maps if metric_maps is None else metric_maps, include_aupro=evaluation_split=="test"))
        if evaluation_split=="test":
            result["image_f1"] = div(2*tp, 2*tp+fp+fn) or 0.0
    return result

def metrics(rows, maps, threshold, evaluation_split=None, calibration=None):
    inputs = {}
    if calibration is not None:
        scores, normalized, pred_labels, pred_masks = post_process(maps, calibration)
        # anomalib evaluates validation AUROC before post-processing; test uses the normalized output.
        inputs = dict(image_scores=maps.max(axis=(1,2)) if evaluation_split=="val" else scores,
                      pred_labels=pred_labels, pred_masks=pred_masks,
                      metric_maps=maps if evaluation_split=="val" else normalized)
    overall = summarize(rows,maps,threshold,evaluation_split,**inputs)
    by = {}
    for eq in sorted({r["equipment"] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r["equipment"]==eq]
        by[eq] = summarize([rows[i] for i in idx], maps[idx], threshold,evaluation_split,
                           **{k:v[idx] for k,v in inputs.items()})
    overall["by_equipment"] = by
    overall["worst_equipment_fpr"] = max((v["fpr"] for v in by.values() if v["fpr"] is not None),default=None)
    return overall

def selection_key(m):
    # Threshold already enforces validation image recall; favor region recall then overkill
    # through explicit small-object recall floor in the outer rank.
    return (m["fpr"], -m["dice"] if m["dice"] is not None else 0)

def train_cnn(exp, parts, args, device, out):
    import copy
    idx = EXPS.index(exp)
    donors = [r for r in parts["train"] if r["label"]==1]
    normals = [r for r in parts["train"] if r["label"]==0]
    unlabeled = [r for r in parts["train"] if r["label"]==-1]
    if idx == 7 and not unlabeled:
        raise SkipExperiment("No train rows with label=-1")
    # Exp1 uses positive images and their background pixels only.
    train = donors + (normals if idx >= 2 else [])
    ds = Samples(train, donors, paste=idx>=4, aug=idx>=5)
    counts = {k:sum(r["label"]==k for r in train) for k in (0,1)}
    weights = [1/counts[r["label"]] for r in train]
    sampler = WeightedRandomSampler(weights, max(len(train), args.batch*args.min_steps), replacement=True)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    model = VoidNet(mix=idx>=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    teacher = copy.deepcopy(model).eval() if idx==7 else None
    best, best_state, history = None, None, []
    pseudo_pos_total = 0
    best_epoch = None
    train_start = time.perf_counter()
    LOG.info("[Train] normal=%d void=%d unlabeled=%d sampled_per_epoch=%d batches=%d epochs=%d",
             counts[0], counts[1], len(unlabeled), len(sampler), len(loader), args.epochs)
    label=exp.split("_",1)[0].upper()
    val_batches=math.ceil(len(parts["val"])/args.batch)
    with stage_progress(f"{label} train + validation", args.epochs*(len(loader)+val_batches)) as train_bar:
        for epoch in range(args.epochs):
            train_bar.set_description_str(f"{label} train E{epoch+1}/{args.epochs}")
            epoch_start = time.perf_counter()
            ds.paste_attempts = ds.paste_successes = ds.aug_count = 0
            mix_before = getattr(model.mix, "applied", 0)
            pseudo_before = pseudo_pos_total
            model.train()
            running = []
            LOG.info("[Epoch %02d/%02d] start", epoch+1, args.epochs)
            for step,(x,y) in enumerate(loader,1):
                x,y = x.to(device), y.to(device)
                loss = loss_fn(model(x),y,hard=idx>=3)
                if teacher is not None and epoch >= max(1,args.epochs//5):
                    u = np.stack([r["x"] for r in random.choices(unlabeled,k=len(x))])
                    weak = torch.from_numpy(u[:,None]).to(device)
                    strong = torch.from_numpy(np.stack([domain_aug(a) for a in u])[:,None]).to(device)
                    with torch.no_grad():
                        pseudo = teacher(weak).sigmoid()
                    student = model(strong)
                    # Only trust confident pseudo positives and negatives, balance each class.
                    pp,pn = pseudo>=args.pseudo_positive,pseudo<=args.pseudo_negative
                    terms = []
                    if pp.any():
                        pseudo_pos_total += int(pp.sum())
                        terms.append(F.binary_cross_entropy_with_logits(student[pp],torch.ones_like(student[pp])))
                    if pn.any() and pp.any():
                        terms.append(F.binary_cross_entropy_with_logits(student[pn],torch.zeros_like(student[pn])))
                    if terms:
                        loss = loss + .25 * sum(terms)/len(terms)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if teacher is not None:
                    with torch.no_grad():
                        for t,p in zip(teacher.parameters(),model.parameters()):
                            t.mul_(.99).add_(p,alpha=.01)
                running.append(float(loss.detach()))
                train_bar.set_postfix(batch=f"{step}/{len(loader)}", loss=f"{running[-1]:.4f}", refresh=False)
                train_bar.update(1)
                if args.log_every and (step % args.log_every == 0 or step == len(loader)):
                    dt = time.perf_counter() - epoch_start
                    LOG.info("[Batch %d/%d] epoch=%d loss_mean=%.5f elapsed=%s train_epoch_ETA=%s | %s",
                             step, len(loader), epoch+1, float(np.mean(running)), elapsed(dt),
                             elapsed(dt*(len(loader)-step)/step), gpu_memory(device))
            train_bar.set_description_str(f"{label} validation E{epoch+1}/{args.epochs}")
            train_bar.set_postfix(refresh=False)
            LOG.info("[Validation] epoch=%d images=%d", epoch+1, len(parts["val"]))
            vm = cnn_maps(model,parts["val"],args,device,batch_done=lambda:train_bar.update(1))
            th = threshold_for(parts["val"],vm,args.recall,args.region_recall)
            met = metrics(parts["val"],vm,th)
            key = rank_key(met,args)
            # Exp7 checkpoint must actually include a pseudo-label update.
            updated = (idx != 7 or pseudo_pos_total > 0) and (best is None or key < best)
            if updated:
                best,best_state = key,copy.deepcopy(model.state_dict())
                best_epoch = epoch+1
            dt = time.perf_counter() - epoch_start
            total = time.perf_counter() - train_start
            eta = total/(epoch+1)*(args.epochs-epoch-1)
            counters = dict(paste_attempts=ds.paste_attempts, paste_successes=ds.paste_successes,
                            domain_aug_samples=ds.aug_count,
                            mixstyle_calls=getattr(model.mix, "applied", 0)-mix_before,
                            pseudo_positive_pixels=pseudo_pos_total-pseudo_before)
            history.append(dict(epoch=epoch+1, loss=float(np.mean(running)), val=met, threshold=th,
                                seconds=dt, best_updated=updated, **counters))
            (out/"history.json").write_text(json.dumps({"epochs":history,
                  "pseudo_positive_pixels":pseudo_pos_total,"best_epoch":best_epoch},indent=2),encoding="utf-8")
            LOG.info("[Epoch %02d/%02d] loss=%.5f | %s | %s | elapsed=%s experiment_train_ETA=%s",
                     epoch+1,args.epochs,float(np.mean(running)),metric_line(met,th),
                     "BEST UPDATED" if updated else f"best_epoch={best_epoch}",elapsed(dt),elapsed(eta))
            if idx>=4:
                LOG.info("[Aug] copy_paste=%d/%d domain_aug=%d mixstyle_calls=%d pseudo_positive_pixels=%d",
                         counters["paste_successes"],counters["paste_attempts"],
                         counters["domain_aug_samples"],counters["mixstyle_calls"],counters["pseudo_positive_pixels"])
            LOG.info("[Memory] %s",gpu_memory(device))
            train_bar.set_postfix(loss=f"{np.mean(running):.4f}", val_fpr=pct(met["fpr"]), val_recall=pct(met["recall"]))
            epoch_note("  %s E%d/%d | loss=%.4f | val FPR=%s recall=%s", label,epoch+1,args.epochs,
                       float(np.mean(running)),pct(met["fpr"]),pct(met["recall"]))
    if idx == 7 and pseudo_pos_total == 0:
        raise SkipExperiment("No confident positive pseudo-labels accepted; semi-supervision not exercised")
    model.load_state_dict(best_state)
    torch.save({"state_dict":model.state_dict(),"mix":idx>=6},out/"model.pt")
    LOG.info("[Checkpoint] selected_epoch=%d saved model.pt", best_epoch)
    return lambda rows, desc=None: cnn_maps(model,rows,args,device,desc=desc)

def rank_key(m,args):
    # Region hit recall is distinct from image recall. Prioritize floor violations.
    small = [m["regions"][k]["recall"] for k in ("1px","2-4px")
             if m["regions"][k]["recall"] is not None]
    deficit = max(0,args.region_recall-min(small)) if small else 0
    return (deficit, m["worst_equipment_fpr"] if m["worst_equipment_fpr"] is not None else 1,
            m["fpr"], -(m["dice"] or 0))

def padim_preprocess(x,size=224,upsample_factor=None):
    from torchvision.transforms import v2
    if isinstance(x, (list, tuple)):
        # Keep each decoded image's integer bit depth through Resize (even in mixed-bit-depth batches).
        pairs = [padim_preprocess(im,size,upsample_factor) for im in x]
        return torch.stack([im for im,_ in pairs]),pairs[0][1]
    x = v2.ToImage()(x)
    native_h,native_w=x.shape[-2:]
    if upsample_factor is not None:
        h,w=native_h*upsample_factor,native_w*upsample_factor
        top,left=0,0
        x=v2.Resize((h,w),antialias=True)(x)
    else:
        scale=size/max(native_h,native_w)
        h,w=round(native_h*scale),round(native_w*scale)
        x=v2.Resize((h,w),antialias=True)(x)
        top,left=(size-h)//2,(size-w)//2
    x=v2.ToDtype(torch.float32,scale=True)(x)
    if upsample_factor is None:
        x=F.pad(x,(left,size-w-left,top,size-h-top),mode="replicate")
    x=x.repeat_interleave(3,dim=-3)  # grayscale -> RGB, no ImageNet mean/std
    return x,(top,left,h,w)

def padim_input(row, transformed=None):
    """Decoded raw integer pixels, or a photometrically transformed sample in the original bit depth."""
    raw = row.get("raw", row["x"])
    if transformed is not None:
        raw = (np.rint(np.clip(transformed,0,1)*np.iinfo(raw.dtype).max).astype(raw.dtype)
               if np.issubdtype(raw.dtype,np.integer) else transformed.astype(np.float32))
    return torch.from_numpy(raw[None])

def create_padim_backbone(weights=None, *, pretrained=True):
    import timm
    if weights:
        # torchvision's original ResNet18 has compatible shapes but different pretrained parameters.
        digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if digest.startswith("f37072fd") or Path(weights).name == "resnet18-f37072fd.pth":
            raise ValueError("torchvision ResNet18 weights are not resnet18.a1_in1k; download the timm checkpoint")
    return timm.create_model(PADIM_BACKBONE, pretrained=pretrained and not weights, checkpoint_path=str(weights or ""))

def restore_padim_map(score,crop,native,size=224,upsample_factor=None):
    if upsample_factor is not None:
        # No padding; avoid a redundant upsample/downsample that blurs tiny peaks.
        if tuple(score.shape[-2:])==tuple(native):
            return score
        return F.interpolate(score,size=native,mode="bilinear",align_corners=False)
    top,left,h,w=crop
    score=F.interpolate(score,size=(size,size),mode="bilinear",align_corners=False)
    return F.interpolate(score[:,:,top:top+h,left:left+w],size=native,mode="bilinear",align_corners=False)

class Padim:
    """Independent ResNet18 PaDiM-style baseline, not the user's deployed checkpoint."""
    def __init__(self,args,device):
        self.device,self.args = device,args
        self.net = create_padim_backbone(args.padim_weights)
        self.net.eval().to(device)
        self.depth=getattr(args,"max_layer_depth",3)
        self.upsample_factor=getattr(args,"upsample_factor",None)
        total=sum((64,128,256)[:self.depth])
        if not 1<=args.padim_dim<=total:
            raise ValueError(f"padim_dim must be <= {total} for depth {self.depth}")
        self.channels = torch.randperm(total,generator=torch.Generator().manual_seed(args.seed))[:args.padim_dim].to(device)
    @torch.no_grad()
    def features(self,x):
        x,crop=padim_preprocess(x,self.args.padim_size,self.upsample_factor)
        x=x.to(self.device)
        n=self.net
        a=n.layer1(n.maxpool(n.act1(n.bn1(n.conv1(x)))))
        feats=[a]
        for i in range(2,self.depth+1):
            feats.append(getattr(n,f"layer{i}")(feats[-1]))
        f=torch.cat([a]+[F.interpolate(v,size=a.shape[-2:],mode="nearest") for v in feats[1:]],1)[:,self.channels]
        self.shape=f.shape[-2:]
        return f.flatten(2).transpose(1,2),crop
    @torch.no_grad()
    def fit(self,rows,desc=None):
        # Streaming first and second moments, avoiding all N feature maps in RAM.
        n=0; s=q=None
        with stage_progress(desc,len(rows),unit="img") as bar:
            for start in range(0,len(rows),self.args.batch):
                x=[padim_input(r) for r in rows[start:start+self.args.batch]]
                f,_=self.features(x)
                f=f.double()
                a=f.sum(0); b=torch.einsum("bpc,bpd->pcd",f,f)
                s=a if s is None else s+a
                q=b if q is None else q+b
                n+=len(f)
                bar.update(len(f))
                if self.args.log_every and (n == len(rows) or math.ceil(n/self.args.batch)%self.args.log_every == 0):
                    LOG.info("[PaDiM fit] normals=%d/%d | %s", n,len(rows),gpu_memory(self.device))
            if desc:
                bar.set_description_str(f"{desc} covariance")
            LOG.info("[PaDiM fit] calculating covariance / Cholesky")
            if n<2: raise ValueError("PaDiM requires >=2 normal training images")
            mu=s/n
            cov=(q-n*mu[:,:,None]*mu[:,None,:])/(n-1)
            cov+=.01*torch.eye(cov.shape[-1],device=self.device,dtype=cov.dtype)
            self.mu=mu.float()
            self.chol=torch.linalg.cholesky(cov).float()
    @torch.no_grad()
    def maps(self,rows,desc=None):
        out=[]
        with stage_progress(desc,len(rows),unit="img") as bar:
            for start in range(0,len(rows),self.args.batch):
                x=[padim_input(r) for r in rows[start:start+self.args.batch]]
                f,(top,left,h,w)=self.features(x)
                d=(f-self.mu).unsqueeze(-1)
                z=torch.linalg.solve_triangular(self.chol,d,upper=False)
                score=z.square().sum((-1,-2)).sqrt().reshape(len(x),1,*self.shape)
                score=restore_padim_map(score,(top,left,h,w),x[0].shape[-2:],self.args.padim_size,self.upsample_factor)
                out.extend(ndimage.gaussian_filter(a, self.args.padim_smooth) for a in score[:,0].cpu().numpy())
                bar.update(len(x))
        return np.asarray(out)

class SkipExperiment(Exception):
    pass

def save_eval(rows,maps,calibration,out,split, *, save_raw_maps=False, panel_scale=4,
              collect_comparison=False, desc=None):
    dest=out/split
    dest.mkdir(exist_ok=True)
    scores, normalized, pred_labels, _=post_process(maps,calibration)
    # Save a single visual artifact per image. Numeric maps are opt-in for diagnostics.
    evidence = []
    if collect_comparison:
        if split != "test":
            raise ValueError("Paired comparison evidence is test-only")
        from comparison_metrics import sample_evidence, write_evidence
    with stage_progress(desc,len(rows),unit="img") as bar, (dest/"scores.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["id","image","equipment","lot","label","score","prediction","raw_score","wafer"])
        for r,m,norm,score,flag in zip(rows,maps,normalized,scores,pred_labels):
            writer.writerow([r["id"],r["image"],r["equipment"],r["lot"],r["label"],float(score),int(flag),float(m.max()),r.get("wafer", "")])
            if save_raw_maps:
                np.save(dest/(r["id"]+".npy"),m)
                np.save(dest/(r["id"]+"_normalized.npy"),norm)
            save_panel(r["x"],norm,NORMALIZED_THRESHOLD,dest/(r["id"]+"_panel.png"),
                       metadata=panel_metadata_lines(r, flag, score), scale=panel_scale)
            if collect_comparison:
                evidence.append(sample_evidence(r, norm, flag, NORMALIZED_THRESHOLD))
            bar.update(1)
    if collect_comparison:
        write_evidence(out/"comparison_stats.json", evidence)

def report(results,out,args):
    (out/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    ok=[r for r in results if r["status"]=="ok"]
    best=min(ok,key=lambda r:rank_key(r["val"],args))["experiment"] if ok else None
    fields=["experiment","status","threshold","seconds","val_image_auroc","val_pixel_auroc","test_image_f1","test_pixel_aupro","test_pixel_auroc","val_fpr","val_recall","val_dice","test_fpr",
            "test_recall","test_dice","test_iou","test_worst_fpr","test_1px","test_2-4px","test_5+px","error"]
    flat=[]
    for r in results:
        row={k:r.get(k) for k in fields}
        if r["status"]=="ok":
            for s in ("val","test"):
                for k in ("fpr","recall","dice","iou"):
                    row[s+"_"+k]=r[s][k]
            for split, keys in (("val", ("image_auroc","pixel_auroc")), ("test", ("image_f1","pixel_aupro","pixel_auroc"))):
                for key in keys: row[split+"_"+key]=r[split].get(key)
            row["test_worst_fpr"]=r["test"]["worst_equipment_fpr"]
            for k in ("1px","2-4px","5+px"): row["test_"+k]=r["test"]["regions"][k]["recall"]
        flat.append(row)
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(flat)
    esc=lambda v:html.escape("" if v is None else f"{v:.4f}" if isinstance(v,float) else str(v))
    cols=fields[:-1]
    body=["<meta charset='utf-8'><title>CSAM experiments</title><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}th{background:#eee}pre{white-space:pre-wrap}</style>",
          "<h1>CSAM Void Segmentation</h1>",f"<p>Validation-selected candidate: {esc(best)}. Test is report-only. Image recall goal: {args.recall}; small-region recall goal: {args.region_recall}. Final decisions use separate F1-adaptive normalization and threshold &gt; 0.5; recall goals are not guaranteed. Lower FPR is better.</p>",
          "<table><tr>"+"".join(f"<th>{esc(k)}</th>" for k in cols)+"</tr>"]
    for row in flat:
        body.append("<tr>"+"".join(f"<td>{esc(row.get(k))}</td>" for k in cols)+"</tr>")
    body.append("</table>")
    for r in results:
        name=r["experiment"]
        body.append(f"<h2>{esc(name)}</h2>")
        if r["status"]!="ok":
            body.append(f"<p>{esc(r.get('error'))}</p>"); continue
        body.append("<table><tr><th>Split</th><th>Equipment</th><th>FP</th><th>FN</th><th>FPR</th><th>Recall</th><th>Dice</th></tr>")
        for s in ("val","test"):
            for eq,m in r[s]["by_equipment"].items():
                body.append("<tr>"+"".join(f"<td>{esc(v)}</td>" for v in (s,eq,m["fp"],m["fn"],m["fpr"],m["recall"],m["dice"]))+"</tr>")
        body.append(f"</table><p><a href='{name}/test/scores.csv'>Test scores</a> | <a href='{name}/test/'>3-panel PNGs</a></p>")
    (out/"report.html").write_text("\n".join(body),encoding="utf-8")
    (out/"selected.json").write_text(json.dumps({"selected_by_validation":best},indent=2),encoding="utf-8")

def run_protocol(parts,args,device,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    (out/"split_audit.json").write_text(json.dumps({s:[{k:r[k] for k in ("id","image","label","equipment","lot","wafer")} for r in rr] for s,rr in parts.items()},indent=2),encoding="utf-8")
    results=[]
    log_dataset(parts)
    console_info("[Data] train=%d | validation=%d | test=%d",
                 len(parts["train"]),len(parts["val"]),len(parts["test"]))
    protocol_start=time.perf_counter()
    for exp_num,exp in enumerate(args.experiments,1):
        start=time.perf_counter()
        seed_all(args.seed)
        dest=out/exp
        dest.mkdir()
        console_info("[Experiment %d/%d] %s / %s", exp_num,len(args.experiments),out.name,exp)
        label=exp.split("_",1)[0].upper()
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        try:
            if exp==EXPS[0] or getattr(args,"padim_suite",False):
                LOG.info("[PaDiM] square_size=%s upsample_factor=%s depth=%d channels=%d normal-only fit",args.padim_size,args.upsample_factor,args.max_layer_depth,args.padim_dim)
                model=Padim(args,device)
                if getattr(args,"padim_suite",False):
                    from padim_backbone import adapt_backbone
                    adapt_backbone(model.net,exp,parts,args,device,dest)
                model.fit([r for r in parts["train"] if r["label"]==0],desc=f"{label} Gaussian fit")
                checkpoint=dict(mu=model.mu.cpu(),chol=model.chol.cpu(),channels=model.channels.cpu(),
                                size=args.padim_size,upsample_factor=args.upsample_factor,max_layer_depth=args.max_layer_depth,n_features=args.padim_dim,feature_shape=tuple(model.shape),smooth=args.padim_smooth,native_shape=args.native_shape,backbone_name=PADIM_BACKBONE,backbone={k:v.cpu() for k,v in model.net.state_dict().items()})
                predict=model.maps
            else:
                predict=train_cnn(exp,parts,args,device,dest)
                checkpoint=torch.load(dest/"model.pt",map_location="cpu",weights_only=True)
            LOG.info("[Final validation] images=%d",len(parts["val"]))
            vm=predict(parts["val"],desc=f"{label} validation")
            if not np.isfinite(vm).all():
                raise ValueError("Nonfinite validation predictions")
            calibration=calibrate_post_processor(parts["val"],vm)
            th=NORMALIZED_THRESHOLD
            checkpoint["post_processor"]=calibration
            torch.save(checkpoint,dest/"model.pt")
            (dest/"threshold.json").write_text(json.dumps(calibration,indent=2,allow_nan=False),encoding="utf-8")
            LOG.info("[Post processor] validation image threshold=%.6g min=%.6g max=%.6g | pixel threshold=%.6g min=%.6g max=%.6g",
                     calibration["image_threshold"],calibration["image_min"],calibration["image_max"],
                     calibration["pixel_threshold"],calibration["pixel_min"],calibration["pixel_max"])
            LOG.info("[Test] images=%d normalized threshold=%.6g (fixed from validation)",len(parts["test"]),th)
            tm=predict(parts["test"],desc=f"{label} test")
            if not np.isfinite(vm).all() or not np.isfinite(tm).all():
                raise ValueError("Nonfinite predictions")
            with stage_progress(f"{label} metrics",2,unit="split") as bar:
                val=metrics(parts["val"],vm,th,"val",calibration)
                bar.update(1)
                test=metrics(parts["test"],tm,th,"test",calibration)
                bar.update(1)
            LOG.info("[Save] val/test 3-panel PNGs and CSV%s", " + diagnostic NPY maps" if args.save_raw_maps else "")
            save_eval(parts["val"],vm,calibration,dest,"val",save_raw_maps=args.save_raw_maps,
                      panel_scale=args.panel_scale,desc=f"{label} save val")
            save_eval(parts["test"],tm,calibration,dest,"test",save_raw_maps=args.save_raw_maps,
                      panel_scale=args.panel_scale, collect_comparison=getattr(args,"padim_suite",False),
                      desc=f"{label} save test")
            result=dict(experiment=exp,status="ok",threshold=th,post_processor=calibration,val=val,test=test)
            if getattr(args,"padim_suite",False):
                from padim_backbone import operating_points
                # Retain the original shared-raw-threshold sweep as a diagnostic, independent of the calibrated decisions.
                points=operating_points(parts["val"],vm,parts["test"],tm,None)
                result["operating_points"]=points
                (dest/"operating_points.json").write_text(json.dumps(points,indent=2,allow_nan=False),encoding="utf-8")
                if (dest/"backbone_summary.json").is_file():
                    result["backbone"]=json.loads((dest/"backbone_summary.json").read_text(encoding="utf-8"))
                for point in points["points"]:
                    if point["kind"]=="image_recall" and point["target"]==.95:
                        LOG.info("[Image-only 95%% validation recall] val=%s test=%s",point["val"],point["test"])
                    if point["kind"]=="fpr_ceiling" and point["target"]==.05:
                        LOG.info("[Validation FPR <=5%%] val=%s test=%s",point["val"],point["test"])
        except SkipExperiment as e:
            result=dict(experiment=exp,status="skipped",error=str(e))
        except Exception as e:
            import traceback
            (dest/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            result=dict(experiment=exp,status="failed",error=str(e))
        result["seconds"]=time.perf_counter()-start
        results.append(result)
        report(results,out,args)
        if result["status"]=="ok":
            LOG.info("[Final Val] %s",metric_line(result["val"],result["threshold"]))
            LOG.info("[Final Test] %s",metric_line(result["test"],result["threshold"]))
            LOG.info("[Standard metrics] val image AUROC=%s pixel AUROC=%s | test image F1=%s pixel AUPRO@0.30=%s pixel AUROC=%s",
                     val["image_auroc"],val["pixel_auroc"],test["image_f1"],test["pixel_aupro"],test["pixel_auroc"])
            console_info("  %s DONE (%s) | Val AUROC image/pixel=%s/%s | Test F1=%s AUPRO=%s pixel AUROC=%s",
                         label,elapsed(result["seconds"]),number(val["image_auroc"]),number(val["pixel_auroc"]),
                         number(test["image_f1"]),number(test["pixel_aupro"]),number(test["pixel_auroc"]))
            console_info("  Test FPR=%s | Recall=%s | 1px=%s | 2-4px=%s | worst FPR=%s",
                         pct(test["fpr"]),pct(test["recall"]),pct(test["regions"]["1px"]["recall"]),
                         pct(test["regions"]["2-4px"]["recall"]),pct(test["worst_equipment_fpr"]))
        else:
            LOG.warning("[%s] %s: %s",result["status"].upper(),exp,result["error"])
        dt=time.perf_counter()-protocol_start
        LOG.info("[Experiment %d/%d] %s %s elapsed=%s protocol_ETA_rough=%s",
                 exp_num,len(args.experiments),exp,result["status"],elapsed(result["seconds"]),
                 elapsed(dt/exp_num*(len(args.experiments)-exp_num)))
        if device.type=="cuda": torch.cuda.empty_cache()
    valid=[r for r in results if r["status"]=="ok"]
    selected=min(valid,key=lambda r:rank_key(r["val"],args))["experiment"] if valid else "none"
    console_info("[Selected by validation] %s / %s",out.name,selected)
    return results

def main(padim_suite=False):
    if padim_suite:
        from padim_backbone import PEXPS
    choices=PEXPS if padim_suite else EXPS
    p=argparse.ArgumentParser(description="PaDiM backbone adaptation P0-P5" if padim_suite else __doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=10 if padim_suite else 30)
    p.add_argument("--log-every",type=int,default=50,help="Detailed run.log batch interval; 0 disables batch log entries, not progress bars")
    p.add_argument("--progress",choices=["auto","on","off"],default="auto",
                   help="auto: tqdm on interactive terminals; on: force bars; off: compact text")
    p.add_argument("--verbose",action="store_true",help="Show detailed run.log messages in the console as well")
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--min-steps",type=int,default=20)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--pseudo-positive",type=float,default=.98)
    p.add_argument("--pseudo-negative",type=float,default=.02)
    p.add_argument("--recall",type=float,default=.95)
    p.add_argument("--region-recall",type=float,default=.90)
    p.add_argument("--device",default="auto")
    geometry=p.add_mutually_exclusive_group()
    geometry.add_argument("--padim-size",type=int,default=None,help="Square letterbox size (legacy mode; overrides suite default factor)")
    geometry.add_argument("--upsample-factor","--upsample_factor",type=int,default=None,help="Integer native H/W scale, no padding")
    p.add_argument("--max-layer-depth","--max_layer_depth",type=int,choices=[1,2,3],default=2)
    p.add_argument("--padim-dim","--n-features","--n_features",dest="padim_dim",type=int,default=40)
    p.add_argument("--padim-smooth",type=float,default=0.)
    p.add_argument("--padim-weights",help="Official local timm resnet18.a1_in1k checkpoint (.safetensors or .pth/.bin); otherwise download it via timm")
    p.add_argument("--save-raw-maps",action="store_true",help="Optionally save raw and normalized .npy maps for numerical diagnostics; default saves only 3-panel PNGs and scores.csv per split")
    p.add_argument("--panel-scale",type=int,default=4,help="Integer display enlargement (1-16), nearest-neighbor; metrics remain at native resolution")
    p.add_argument("--experiments",nargs="+",choices=choices,default=choices)
    if padim_suite:
        p.add_argument("--backbone-lr",type=float,default=1e-5)
        p.add_argument("--head-lr",type=float,default=1e-3)
        p.add_argument("--mix-prob",type=float,default=.5)
        p.add_argument("--mix-alpha",type=float,default=.1)
        p.add_argument("--pixel-weight",type=float,default=1.)
        p.add_argument("--anchor-weight",type=float,default=1.)
    p.add_argument("--protocol",choices=["pooled","leave-one-equipment-out","both"],default="pooled")
    args=p.parse_args()
    args.padim_suite=padim_suite
    if args.padim_size is None and args.upsample_factor is None:
        args.upsample_factor=4
    if args.upsample_factor is not None and args.upsample_factor<1:
        p.error("upsample-factor must be >= 1")
    if padim_suite and not (args.backbone_lr>0 and args.head_lr>0 and 0<args.mix_prob<=1 and args.mix_alpha>0 and args.pixel_weight>0 and args.anchor_weight>0):
        p.error("Invalid backbone adaptation hyperparameters")
    if not (0<args.recall<=1 and 0<=args.region_recall<=1 and args.epochs>0 and args.batch>1 and args.min_steps>0 and 1<=args.padim_dim<=sum((64,128,256)[:args.max_layer_depth]) and (args.padim_size is None or args.padim_size>=32) and args.lr>0 and args.padim_smooth>=0):
        p.error("Invalid hyperparameters")
    if not (0 <= args.pseudo_negative < args.pseudo_positive <= 1):
        p.error("Require 0 <= pseudo-negative < pseudo-positive <= 1")
    if padim_suite and not args.padim_weights:
        p.error("P0-P5 require --padim-weights with the local timm resnet18.a1_in1k checkpoint (same starting weights for all runs)")
    if args.log_every < 0:
        p.error("--log-every must be >= 0")
    if not 1 <= args.panel_scale <= 16:
        p.error("--panel-scale must be an integer from 1 to 16")
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    setup_logging(root,progress_mode=args.progress,verbose=args.verbose)
    console_info("[Dataset] loading and auditing manifest")
    try:
        rows=load_manifest(args.manifest)
    except Exception:
        LOG.exception("[Dataset] audit failed")
        raise
    args.native_shape=list(rows[0]["x"].shape)
    args.padim_backbone=PADIM_BACKBONE
    LOG.info("Native image shape (H,W): %s",tuple(args.native_shape))
    input_shape=tuple(v*args.upsample_factor for v in args.native_shape) if args.upsample_factor is not None else (args.padim_size,)*2
    args.padim_input_shape=list(input_shape)
    LOG.info("[PaDiM config] input=%s factor=%s depth=%d n_features=%d padding=%s",input_shape,args.upsample_factor,args.max_layer_depth,args.padim_dim,args.upsample_factor is None)
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    LOG.info("[Device] %s | %s",torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",gpu_memory(device))
    LOG.info("[Config] epochs=%d batch=%d lr=%g seed=%d recall=%s region_recall=%s",
             args.epochs,args.batch,args.lr,args.seed,pct(args.recall),pct(args.region_recall))
    console_info("[Run] %s | %s | native=%sx%s | input=%sx%s | epochs=%d batch=%d",
                 "P0-P5" if padim_suite else "Exp0-7",device,
                 *args.native_shape,*args.padim_input_shape,args.epochs,args.batch)
    protocols=[]
    if args.protocol in ("pooled","both"): protocols.append(("pooled",None))
    if args.protocol in ("leave-one-equipment-out","both"):
        protocols += [(f"holdout_{i:02d}",eq) for i,eq in enumerate(sorted({r["equipment"] for r in rows}),1)]
    index=[]
    for proto_num,(name,eq) in enumerate(protocols,1):
        console_info("[Protocol %d/%d] %s heldout_equipment=%s",proto_num,len(protocols),name,eq or "none")
        try:
            parts=split_rows(rows,eq)
        except ValueError as e:
            LOG.warning("[Protocol skipped] %s: %s",name,e)
            index.append(dict(protocol=name,equipment=eq,status="skipped",error=str(e)))
            continue
        results=run_protocol(parts,args,device,root/name)
        status="ok" if all(r["status"]=="ok" for r in results) else "partial"
        index.append(dict(protocol=name,equipment=eq,status=status))
    (root/"index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    from build_report import build_report
    if args.padim_suite:
        from build_comparison_report import build_comparison_report
        with stage_progress("Build comparison report",1,unit="report") as bar:
            build_comparison_report(root,index)
            bar.update(1)
        console_info("PaDiM paired comparison: %s",root/"comparison.html")
    with stage_progress("Build experiment report",1,unit="report") as bar:
        build_report(root,index)
        bar.update(1)
    console_info("Report: %s",root/"index.html")
    console_info("Log: %s",root/"run.log")
    if any(r["status"]!="ok" for r in index):
        raise SystemExit(2)

if __name__=="__main__":
    main()
