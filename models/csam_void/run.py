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
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

H, W = 61, 81
EXPS = ["exp0_padim", "exp1_cnn", "exp2_normals", "exp3_hard_negative",
        "exp4_copy_paste", "exp5_domain_aug", "exp6_mixstyle", "exp7_semi"]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_image(path, mask=False):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.shape != (H, W):
        raise ValueError(f"{path}: expected grayscale {H}x{W}, got {a.shape}")
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
            x = read_image(p)
            m = read_image((path.parent / r["mask"]).resolve(), True) if r["mask"].strip() else np.zeros_like(x)
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
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x, y = r["x"].copy(), r["y"].copy()
        if self.paste and r["label"] == 0 and self.donors and random.random() < .5:
            pair = copy_void(x, random.choice(self.donors))
            if pair is not None:
                x, y = pair
        if self.aug:
            x = domain_aug(x)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])

class MixStyle(nn.Module):
    def __init__(self, p=.5, alpha=.1):
        super().__init__()
        self.p, self.alpha = p, alpha
    def forward(self, x):
        if not self.training or len(x) < 2 or random.random() >= self.p:
            return x
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

def threshold_for(rows, maps, recall):
    positive = sorted((float(m.max()) for r,m in zip(rows,maps) if r["label"] == 1), reverse=True)
    if not positive:
        raise ValueError("Validation has no positive images")
    return positive[max(0, math.ceil(recall * len(positive))-1)]

def summarize(rows, maps, threshold):
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
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, recall=div(tp,tp+fn), fpr=div(fp,fp+tn),
                underkill=div(fn,tp+fn), dice=div(2*inter,predpix+truepix), iou=div(inter,union),
                auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                fp_regions_per_normal=div(false_regions,fp+tn),
                regions={k:dict(hit=v[0], count=v[1], recall=div(*v)) for k,v in region.items()})

def metrics(rows, maps, threshold):
    overall = summarize(rows,maps,threshold)
    by = {}
    for eq in sorted({r["equipment"] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r["equipment"]==eq]
        by[eq] = summarize([rows[i] for i in idx], maps[idx], threshold)
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
    for epoch in range(args.epochs):
        model.train()
        running = []
        for x,y in loader:
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
                pp,pn = pseudo>=.98,pseudo<=.02
                terms = []
                if pp.any():
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
        vm = cnn_maps(model,parts["val"],args,device)
        th = threshold_for(parts["val"],vm,args.recall)
        met = metrics(parts["val"],vm,th)
        key = rank_key(met,args)
        history.append(dict(epoch=epoch+1, loss=float(np.mean(running)), val=met, threshold=th))
        if best is None or key < best:
            best,best_state = key,copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save({"state_dict":model.state_dict(),"mix":idx>=6},out/"model.pt")
    (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return lambda rows: cnn_maps(model,rows,args,device)

def rank_key(m,args):
    # Region hit recall is distinct from image recall. Prioritize floor violations.
    small = [m["regions"][k]["recall"] for k in ("1px","2-4px")
             if m["regions"][k]["recall"] is not None]
    deficit = max(0,args.region_recall-min(small)) if small else 0
    return (deficit, m["worst_equipment_fpr"] if m["worst_equipment_fpr"] is not None else 1,
            m["fpr"], -(m["dice"] or 0))

class Padim:
    """Independent ResNet18 PaDiM-style baseline, not the user's deployed checkpoint."""
    def __init__(self,args,device):
        from torchvision.models import resnet18, ResNet18_Weights
        self.device,self.args = device,args
        self.net = resnet18(weights=None if args.padim_weights else ResNet18_Weights.DEFAULT)
        if args.padim_weights:
            self.net.load_state_dict(torch.load(args.padim_weights,map_location="cpu",weights_only=True))
        self.net.eval().to(device)
        self.channels = torch.randperm(448,generator=torch.Generator().manual_seed(args.seed))[:args.padim_dim].to(device)
    @torch.no_grad()
    def features(self,x):
        x = F.interpolate(x.to(self.device),size=(round(H*self.args.padim_size/W),self.args.padim_size),
                          mode="bilinear",align_corners=False)
        h,w=x.shape[-2:]
        pad=self.args.padim_size-h
        x=F.pad(x,(0,0,pad//2,pad-pad//2),mode="replicate").repeat(1,3,1,1)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        n=self.net
        a=n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x)))))
        b=n.layer2(a); c=n.layer3(b)
        f=torch.cat([a,F.interpolate(b,size=a.shape[-2:],mode="nearest"),
                     F.interpolate(c,size=a.shape[-2:],mode="nearest")],1)[:,self.channels]
        self.shape=f.shape[-2:]
        return f.flatten(2).transpose(1,2), (pad//2,h)
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
            f,(top,h)=self.features(x)
            d=(f-self.mu).unsqueeze(-1)
            z=torch.linalg.solve_triangular(self.chol,d,upper=False)
            score=z.square().sum((-1,-2)).sqrt().reshape(len(x),1,*self.shape)
            score=F.interpolate(score,size=(self.args.padim_size,)*2,mode="bilinear",align_corners=False)
            score=F.interpolate(score[:,:,top:top+h],size=(H,W),mode="bilinear",align_corners=False)
            out.extend(ndimage.gaussian_filter(a, self.args.padim_smooth) for a in score[:,0].cpu().numpy())
        return np.asarray(out)

class SkipExperiment(Exception):
    pass

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
            if flag or r["label"]==1:
                rgb=np.repeat((r["x"]*255).astype(np.uint8)[...,None],3,axis=2)
                rgb[m>=th]=[255,65,65]
                rgb[r["y"]>0]=[65,255,65]
                rgb[(m>=th)&(r["y"]>0)]=[255,255,0]
                Image.fromarray(rgb).resize((W*4,H*4),Image.Resampling.NEAREST).save(dest/(r["id"]+"_overlay.png"))

def report(results,out,args):
    (out/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    ok=[r for r in results if r["status"]=="ok"]
    best=min(ok,key=lambda r:rank_key(r["val"],args))["experiment"] if ok else None
    fields=["experiment","status","threshold","seconds","val_fpr","val_recall","val_dice","test_fpr",
            "test_recall","test_dice","test_iou","test_worst_fpr","test_1px","test_2-4px","test_5+px","error"]
    flat=[]
    for r in results:
        row={k:r.get(k) for k in fields}
        if r["status"]=="ok":
            for s in ("val","test"):
                for k in ("fpr","recall","dice","iou"):
                    row[s+"_"+k]=r[s][k]
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
        body.append(f"</table><p><a href='{name}/test/scores.csv'>Test scores</a> | <a href='{name}/test/'>Maps and overlays</a></p>")
    (out/"report.html").write_text("\n".join(body),encoding="utf-8")
    (out/"selected.json").write_text(json.dumps({"selected_by_validation":best},indent=2),encoding="utf-8")

def run_protocol(parts,args,device,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    (out/"split_audit.json").write_text(json.dumps({s:[{k:r[k] for k in ("id","image","label","equipment","lot")} for r in rr] for s,rr in parts.items()},indent=2),encoding="utf-8")
    results=[]
    for exp in args.experiments:
        start=time.perf_counter()
        seed_all(args.seed)
        dest=out/exp
        dest.mkdir()
        try:
            if exp==EXPS[0]:
                model=Padim(args,device)
                model.fit([r for r in parts["train"] if r["label"]==0])
                torch.save(dict(mu=model.mu.cpu(),chol=model.chol.cpu(),channels=model.channels.cpu(),
                                size=args.padim_size,smooth=args.padim_smooth),dest/"model.pt")
                predict=model.maps
            else:
                predict=train_cnn(exp,parts,args,device,dest)
            vm=predict(parts["val"])
            th=threshold_for(parts["val"],vm,args.recall)
            tm=predict(parts["test"])
            if not np.isfinite(vm).all() or not np.isfinite(tm).all():
                raise ValueError("Nonfinite predictions")
            val,test=metrics(parts["val"],vm,th),metrics(parts["test"],tm,th)
            save_eval(parts["val"],vm,th,dest,"val")
            save_eval(parts["test"],tm,th,dest,"test")
            result=dict(experiment=exp,status="ok",threshold=th,val=val,test=test)
        except SkipExperiment as e:
            result=dict(experiment=exp,status="skipped",error=str(e))
        except Exception as e:
            import traceback
            (dest/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
            result=dict(experiment=exp,status="failed",error=str(e))
        result["seconds"]=time.perf_counter()-start
        results.append(result)
        report(results,out,args)
        print(exp,result["status"],flush=True)
        if device.type=="cuda": torch.cuda.empty_cache()
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--min-steps",type=int,default=20)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--recall",type=float,default=.95)
    p.add_argument("--region-recall",type=float,default=.90)
    p.add_argument("--device",default="auto")
    p.add_argument("--padim-size",type=int,default=224)
    p.add_argument("--padim-dim",type=int,default=100)
    p.add_argument("--padim-smooth",type=float,default=0.)
    p.add_argument("--padim-weights",help="Local torchvision ResNet18 state_dict; otherwise downloads official ImageNet weights")
    p.add_argument("--experiments",nargs="+",choices=EXPS,default=EXPS)
    p.add_argument("--protocol",choices=["pooled","leave-one-equipment-out","both"],default="pooled")
    args=p.parse_args()
    if not (0<args.recall<=1 and 0<=args.region_recall<=1 and args.epochs>0 and args.batch>1 and args.min_steps>0 and 1<=args.padim_dim<=448 and args.padim_size>=32 and args.lr>0 and args.padim_smooth>=0):
        p.error("Invalid hyperparameters")
    rows=load_manifest(args.manifest)
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=False)
    protocols=[]
    if args.protocol in ("pooled","both"): protocols.append(("pooled",None))
    if args.protocol in ("leave-one-equipment-out","both"):
        protocols += [(f"holdout_{i:02d}",eq) for i,eq in enumerate(sorted({r["equipment"] for r in rows}),1)]
    index=[]
    for name,eq in protocols:
        try:
            parts=split_rows(rows,eq)
        except ValueError as e:
            index.append(dict(protocol=name,equipment=eq,status="skipped",error=str(e)))
            continue
        results=run_protocol(parts,args,device,root/name)
        status="ok" if all(r["status"]=="ok" for r in results) else "partial"
        index.append(dict(protocol=name,equipment=eq,status=status))
    (root/"index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    (root/"index.html").write_text("<meta charset='utf-8'><h1>CSAM experiment reports</h1>"+"".join(
        f"<p>{html.escape(str(r['equipment']))}: {html.escape(r['status'])} "+
        (f"<a href='{r['protocol']}/report.html'>{r['protocol']}</a>" if r["status"]!="skipped" else html.escape(r["error"]))+"</p>" for r in index),encoding="utf-8")
    print(f"Report: {root/'index.html'}")
    if any(r["status"]!="ok" for r in index):
        raise SystemExit(2)

if __name__=="__main__":
    main()
