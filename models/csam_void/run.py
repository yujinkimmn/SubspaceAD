"""CSAM experiment runner. Local data only; never uploads images."""
import argparse
import csv
import hashlib
import html
import json
import logging
import sys
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

H, W = 61, 81
EXPS = ["exp0_padim", "exp1_cnn", "exp2_normals", "exp3_hard_negative",
        "exp4_copy_paste", "exp5_domain_aug", "exp6_mixstyle", "exp7_semi"]

LOG = logging.getLogger("csam_void")

def setup_logging(root):
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(root / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)

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

def read_image(path, mask=False, expected_shape=None):
    with Image.open(path) as im:
        a = np.asarray(im)
    allowed = (expected_shape,) if expected_shape is not None else ((H, W), (W, H))
    if a.shape not in allowed:
        raise ValueError(f"{path}: expected grayscale H,W in {allowed}, got {a.shape}")
    if mask:
        return (a > 0).astype(np.float32)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535
    raise ValueError(f"{path}: only uint8/uint16 grayscale supported; got {a.dtype}")

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
            if split not in ("train", "val", "test") or label not in (-1, 0, 1) or not eq or not lot:
                raise ValueError(f"Bad manifest row {i}")
            if label == -1 and split != "train":
                raise ValueError("Unlabeled samples allowed only in train")
            p = (path.parent / r["image"]).resolve()
            x = read_image(p, expected_shape=rows[0]["x"].shape if rows else None)
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
            rows.append(dict(id=str(i), image=str(p), x=x, y=m, label=label,
                             split=split, equipment=eq, lot=lot))
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
def cnn_maps(model, rows, args, device):
    model.eval()
    maps = []
    for x, _ in DataLoader(Samples(rows), batch_size=args.batch):
        maps.extend(model(x.to(device)).sigmoid().cpu().numpy()[:,0])
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


def summarize(rows, maps, threshold, evaluation_split=None):
    tp = fp = tn = fn = inter = union = predpix = truepix = 0
    region = {k:[0,0] for k in ("1px", "2-4px", "5+px")}
    false_regions = 0
    labels, scores = [], []
    for r,m in zip(rows,maps):
        pred = m >= threshold
        actual = r["label"] == 1
        flag = bool(pred.any())
        tp += int(actual and flag); fn += int(actual and not flag)
        fp += int(not actual and flag); tn += int(not actual and not flag)
        labels.append(int(actual)); scores.append(float(m.max()))
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
        result.update(pixel_metrics(rows, maps, include_aupro=evaluation_split=="test"))
        if evaluation_split=="test":
            result["image_f1"] = div(2*tp, 2*tp+fp+fn) or 0.0
    return result

def metrics(rows, maps, threshold, evaluation_split=None):
    overall = summarize(rows,maps,threshold,evaluation_split)
    by = {}
    for eq in sorted({r["equipment"] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r["equipment"]==eq]
        by[eq] = summarize([rows[i] for i in idx], maps[idx], threshold,evaluation_split)
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
    for epoch in range(args.epochs):
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
            if args.log_every and (step % args.log_every == 0 or step == len(loader)):
                dt = time.perf_counter() - epoch_start
                LOG.info("[Batch %d/%d] epoch=%d loss_mean=%.5f elapsed=%s train_epoch_ETA=%s | %s",
                         step, len(loader), epoch+1, float(np.mean(running)), elapsed(dt),
                         elapsed(dt*(len(loader)-step)/step), gpu_memory(device))
        LOG.info("[Validation] epoch=%d images=%d", epoch+1, len(parts["val"]))
        vm = cnn_maps(model,parts["val"],args,device)
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
    if idx == 7 and pseudo_pos_total == 0:
        raise SkipExperiment("No confident positive pseudo-labels accepted; semi-supervision not exercised")
    model.load_state_dict(best_state)
    torch.save({"state_dict":model.state_dict(),"mix":idx>=6},out/"model.pt")
    LOG.info("[Checkpoint] selected_epoch=%d saved model.pt", best_epoch)
    return lambda rows: cnn_maps(model,rows,args,device)

def rank_key(m,args):
    # Region hit recall is distinct from image recall. Prioritize floor violations.
    small = [m["regions"][k]["recall"] for k in ("1px","2-4px")
             if m["regions"][k]["recall"] is not None]
    deficit = max(0,args.region_recall-min(small)) if small else 0
    return (deficit, m["worst_equipment_fpr"] if m["worst_equipment_fpr"] is not None else 1,
            m["fpr"], -(m["dice"] or 0))

def padim_preprocess(x,size=224,upsample_factor=None):
    native_h,native_w=x.shape[-2:]
    if upsample_factor is not None:
        h,w=native_h*upsample_factor,native_w*upsample_factor
        top,left=0,0
        x=F.interpolate(x,size=(h,w),mode="bilinear",align_corners=False)
    else:
        scale=size/max(native_h,native_w)
        h,w=round(native_h*scale),round(native_w*scale)
        x=F.interpolate(x,size=(h,w),mode="bilinear",align_corners=False)
        top,left=(size-h)//2,(size-w)//2
        x=F.pad(x,(left,size-w-left,top,size-h-top),mode="replicate")
    x=x.repeat(1,3,1,1)
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x,(top,left,h,w)

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
        from torchvision.models import resnet18, ResNet18_Weights
        self.device,self.args = device,args
        self.net = resnet18(weights=None if args.padim_weights else ResNet18_Weights.DEFAULT)
        if args.padim_weights:
            self.net.load_state_dict(torch.load(args.padim_weights,map_location="cpu",weights_only=True))
        self.net.eval().to(device)
        self.depth=getattr(args,"max_layer_depth",3)
        self.upsample_factor=getattr(args,"upsample_factor",None)
        total=sum((64,128,256)[:self.depth])
        if not 1<=args.padim_dim<=total:
            raise ValueError(f"padim_dim must be <= {total} for depth {self.depth}")
        self.channels = torch.randperm(total,generator=torch.Generator().manual_seed(args.seed))[:args.padim_dim].to(device)
    @torch.no_grad()
    def features(self,x):
        x,crop=padim_preprocess(x.to(self.device),self.args.padim_size,self.upsample_factor)
        n=self.net
        a=n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x)))))
        feats=[a]
        for i in range(2,self.depth+1):
            feats.append(getattr(n,f"layer{i}")(feats[-1]))
        f=torch.cat([a]+[F.interpolate(v,size=a.shape[-2:],mode="nearest") for v in feats[1:]],1)[:,self.channels]
        self.shape=f.shape[-2:]
        return f.flatten(2).transpose(1,2),crop
    @torch.no_grad()
    def fit(self,rows):
        # Streaming first and second moments, avoiding all N feature maps in RAM.
        n=0; s=q=None
        for x,_ in DataLoader(Samples(rows),batch_size=self.args.batch):
            f,_=self.features(x)
            f=f.double()
            a=f.sum(0); b=torch.einsum("bpc,bpd->pcd",f,f)
            s=a if s is None else s+a
            q=b if q is None else q+b
            n+=len(f)
            if self.args.log_every and (n == len(rows) or math.ceil(n/self.args.batch)%self.args.log_every == 0):
                LOG.info("[PaDiM fit] normals=%d/%d | %s", n,len(rows),gpu_memory(self.device))
        LOG.info("[PaDiM fit] calculating covariance / Cholesky")
        if n<2: raise ValueError("PaDiM requires >=2 normal training images")
        mu=s/n
        cov=(q-n*mu[:,:,None]*mu[:,None,:])/(n-1)
        cov+=.01*torch.eye(cov.shape[-1],device=self.device,dtype=cov.dtype)
        self.mu=mu.float()
        self.chol=torch.linalg.cholesky(cov).float()
    @torch.no_grad()
    def maps(self,rows):
        out=[]
        for x,_ in DataLoader(Samples(rows),batch_size=self.args.batch):
            f,(top,left,h,w)=self.features(x)
            d=(f-self.mu).unsqueeze(-1)
            z=torch.linalg.solve_triangular(self.chol,d,upper=False)
            score=z.square().sum((-1,-2)).sqrt().reshape(len(x),1,*self.shape)
            score=restore_padim_map(score,(top,left,h,w),x.shape[-2:],self.args.padim_size,self.upsample_factor)
            out.extend(ndimage.gaussian_filter(a, self.args.padim_smooth) for a in score[:,0].cpu().numpy())
        return np.asarray(out)

class SkipExperiment(Exception):
    pass

def save_panel(x, score, threshold, path):
    """Three aligned native-coordinate views, with a fixed threshold-relative heat scale."""
    h, w = x.shape
    # Keep raw scores and the actual mask independent of heatmap display clipping.
    vmax = 2 * float(threshold) if threshold > 0 else 1.0
    anchors = np.array([[0, 0, 96], [0, 155, 255], [255, 230, 0], [220, 0, 0]], dtype=np.float32)
    def heat(values):
        t = np.clip(values / vmax, 0, 1)
        return np.stack([np.interp(t, np.linspace(0, 1, len(anchors)), anchors[:, c])
                         for c in range(3)], axis=-1).astype(np.uint8)
    original = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    mask = ((score >= threshold) * 255).astype(np.uint8)
    panels = [Image.fromarray(original).convert("RGB"), Image.fromarray(heat(score)),
              Image.fromarray(mask).convert("RGB")]
    gap, margin, top, footer = 12, 12, 32, 64
    pw, ph = w, h
    canvas = Image.new("RGB", (2 * margin + 3 * pw + 2 * gap, top + ph + footer), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel) in enumerate(zip(("Original", "Anomaly map", "Pred mask"), panels)):
        left = margin + i * (pw + gap)
        draw.text((left + pw // 2, 10), title, anchor="mt", fill="black")
        canvas.paste(panel, (left, top))
    left, y = margin + pw + gap, top + ph + 7
    bar = heat(np.linspace(0, vmax, pw, dtype=np.float64)[None, :])
    canvas.paste(Image.fromarray(bar).resize((pw, 9), Image.Resampling.NEAREST), (left, y))
    tick = left + round(np.clip(threshold / vmax, 0, 1) * (pw - 1))
    draw.line((tick, y - 2, tick, y + 12), fill="black", width=1)
    draw.text((left, y + 15), "0", fill="black")
    draw.text((left + pw, y + 15), f"{vmax:.3g}+", anchor="ra", fill="black")
    draw.text((margin, y + 30), f"Threshold: {threshold:.3g} | Max: {score.max():.3g}", fill="black")
    draw.text((margin, y + 44), "White = predicted defect", fill="black")
    canvas.save(path)


def save_eval(rows,maps,th,out,split):
    dest=out/split
    dest.mkdir(exist_ok=True)
    # Save native score maps and binary masks, plus audit CSV.
    with (dest/"scores.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["id","image","equipment","lot","label","score","prediction"])
        for r,m in zip(rows,maps):
            flag=int(m.max()>=th)
            writer.writerow([r["id"],r["image"],r["equipment"],r["lot"],r["label"],float(m.max()),flag])
            np.save(dest/(r["id"]+".npy"),m)
            Image.fromarray(((m>=th)*255).astype(np.uint8)).save(dest/(r["id"]+"_mask.png"))
            save_panel(r["x"], m, th, dest/(r["id"]+"_panel.png"))
            if flag or r["label"]==1:
                rgb=np.repeat((r["x"]*255).astype(np.uint8)[...,None],3,axis=2)
                rgb[m>=th]=[255,65,65]
                rgb[r["y"]>0]=[65,255,65]
                rgb[(m>=th)&(r["y"]>0)]=[255,255,0]
                Image.fromarray(rgb).resize((rgb.shape[1]*4,rgb.shape[0]*4),Image.Resampling.NEAREST).save(dest/(r["id"]+"_overlay.png"))

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
          "<h1>CSAM Void Segmentation</h1>",f"<p>Validation-selected candidate: {esc(best)}. Test is report-only. Image recall target: {args.recall}; small-region recall floor: {args.region_recall}. Lower FPR is better. Red=prediction, green=GT, yellow=overlap.</p>",
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
        body.append(f"</table><p><a href='{name}/test/scores.csv'>Test scores</a> | <a href='{name}/test/'>Maps, masks and panels</a></p>")
    (out/"report.html").write_text("\n".join(body),encoding="utf-8")
    (out/"selected.json").write_text(json.dumps({"selected_by_validation":best},indent=2),encoding="utf-8")

def run_protocol(parts,args,device,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    (out/"split_audit.json").write_text(json.dumps({s:[{k:r[k] for k in ("id","image","label","equipment","lot")} for r in rr] for s,rr in parts.items()},indent=2),encoding="utf-8")
    results=[]
    log_dataset(parts)
    protocol_start=time.perf_counter()
    for exp_num,exp in enumerate(args.experiments,1):
        start=time.perf_counter()
        seed_all(args.seed)
        dest=out/exp
        dest.mkdir()
        LOG.info("[Experiment %d/%d] %s / %s START", exp_num,len(args.experiments),out.name,exp)
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        try:
            if exp==EXPS[0] or getattr(args,"padim_suite",False):
                LOG.info("[PaDiM] square_size=%s upsample_factor=%s depth=%d channels=%d normal-only fit",args.padim_size,args.upsample_factor,args.max_layer_depth,args.padim_dim)
                model=Padim(args,device)
                if getattr(args,"padim_suite",False):
                    from padim_backbone import adapt_backbone
                    adapt_backbone(model.net,exp,parts,args,device,dest)
                model.fit([r for r in parts["train"] if r["label"]==0])
                torch.save(dict(mu=model.mu.cpu(),chol=model.chol.cpu(),channels=model.channels.cpu(),
                                size=args.padim_size,upsample_factor=args.upsample_factor,max_layer_depth=args.max_layer_depth,n_features=args.padim_dim,feature_shape=tuple(model.shape),smooth=args.padim_smooth,native_shape=args.native_shape,backbone={k:v.cpu() for k,v in model.net.state_dict().items()}),dest/"model.pt")
                predict=model.maps
            else:
                predict=train_cnn(exp,parts,args,device,dest)
            LOG.info("[Final validation] images=%d",len(parts["val"]))
            vm=predict(parts["val"])
            th=threshold_for(parts["val"],vm,args.recall,args.region_recall)
            LOG.info("[Test] images=%d threshold=%.6g (fixed from validation)",len(parts["test"]),th)
            tm=predict(parts["test"])
            if not np.isfinite(vm).all() or not np.isfinite(tm).all():
                raise ValueError("Nonfinite predictions")
            val,test=metrics(parts["val"],vm,th,"val"),metrics(parts["test"],tm,th,"test")
            (dest/"threshold.json").write_text(json.dumps({"threshold":th,"source":"validation"}),encoding="utf-8")
            LOG.info("[Save] val/test maps, masks, overlays, 3-panel PNGs and CSV")
            save_eval(parts["val"],vm,th,dest,"val")
            save_eval(parts["test"],tm,th,dest,"test")
            result=dict(experiment=exp,status="ok",threshold=th,val=val,test=test)
            if getattr(args,"padim_suite",False):
                from padim_backbone import operating_points
                points=operating_points(parts["val"],vm,parts["test"],tm,th)
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
        else:
            LOG.warning("[%s] %s: %s",result["status"].upper(),exp,result["error"])
        dt=time.perf_counter()-protocol_start
        LOG.info("[Experiment %d/%d] %s %s elapsed=%s protocol_ETA_rough=%s",
                 exp_num,len(args.experiments),exp,result["status"],elapsed(result["seconds"]),
                 elapsed(dt/exp_num*(len(args.experiments)-exp_num)))
        if device.type=="cuda": torch.cuda.empty_cache()
    return results

def main(padim_suite=False):
    if padim_suite:
        from padim_backbone import PEXPS
    choices=PEXPS if padim_suite else EXPS
    p=argparse.ArgumentParser(description="PaDiM backbone adaptation P0-P5" if padim_suite else __doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=10 if padim_suite else 30)
    p.add_argument("--log-every",type=int,default=50,help="Batch progress interval; 0 disables batch logs (epoch logs remain)")
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
    p.add_argument("--max-layer-depth","--max_layer_depth",type=int,choices=[1,2,3],default=2 if padim_suite else 3)
    p.add_argument("--padim-dim","--n-features","--n_features",dest="padim_dim",type=int,default=40 if padim_suite else 100)
    p.add_argument("--padim-smooth",type=float,default=0.)
    p.add_argument("--padim-weights",help="Local torchvision ResNet18 state_dict; otherwise downloads official ImageNet weights")
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
        if padim_suite: args.upsample_factor=4
        else: args.padim_size=224
    if args.upsample_factor is not None and args.upsample_factor<1:
        p.error("upsample-factor must be >= 1")
    if padim_suite and not (args.backbone_lr>0 and args.head_lr>0 and 0<args.mix_prob<=1 and args.mix_alpha>0 and args.pixel_weight>0 and args.anchor_weight>0):
        p.error("Invalid backbone adaptation hyperparameters")
    if not (0<args.recall<=1 and 0<=args.region_recall<=1 and args.epochs>0 and args.batch>1 and args.min_steps>0 and 1<=args.padim_dim<=sum((64,128,256)[:args.max_layer_depth]) and (args.padim_size is None or args.padim_size>=32) and args.lr>0 and args.padim_smooth>=0):
        p.error("Invalid hyperparameters")
    if not (0 <= args.pseudo_negative < args.pseudo_positive <= 1):
        p.error("Require 0 <= pseudo-negative < pseudo-positive <= 1")
    if padim_suite and not args.padim_weights:
        p.error("P0-P5 require --padim-weights with a local torchvision ResNet18 state_dict (same starting weights for all runs)")
    if args.log_every < 0:
        p.error("--log-every must be >= 0")
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    setup_logging(root)
    LOG.info("[Dataset] loading and auditing manifest")
    try:
        rows=load_manifest(args.manifest)
    except Exception:
        LOG.exception("[Dataset] audit failed")
        raise
    args.native_shape=list(rows[0]["x"].shape)
    LOG.info("Native image shape (H,W): %s",tuple(args.native_shape))
    input_shape=tuple(v*args.upsample_factor for v in args.native_shape) if args.upsample_factor is not None else (args.padim_size,)*2
    args.padim_input_shape=list(input_shape)
    LOG.info("[PaDiM config] input=%s factor=%s depth=%d n_features=%d padding=%s",input_shape,args.upsample_factor,args.max_layer_depth,args.padim_dim,args.upsample_factor is None)
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    LOG.info("[Device] %s | %s",torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",gpu_memory(device))
    LOG.info("[Config] epochs=%d batch=%d lr=%g seed=%d recall=%s region_recall=%s",
             args.epochs,args.batch,args.lr,args.seed,pct(args.recall),pct(args.region_recall))
    protocols=[]
    if args.protocol in ("pooled","both"): protocols.append(("pooled",None))
    if args.protocol in ("leave-one-equipment-out","both"):
        protocols += [(f"holdout_{i:02d}",eq) for i,eq in enumerate(sorted({r["equipment"] for r in rows}),1)]
    index=[]
    for proto_num,(name,eq) in enumerate(protocols,1):
        LOG.info("[Protocol %d/%d] %s heldout_equipment=%s",proto_num,len(protocols),name,eq or "none")
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
    build_report(root,index)
    LOG.info("Report: %s",root/"index.html")
    LOG.info("Log: %s",root/"run.log")
    if any(r["status"]!="ok" for r in index):
        raise SystemExit(2)

if __name__=="__main__":
    main()
