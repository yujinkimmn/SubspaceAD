"""Train and evaluate NAGL on a single device, with fixed references per equipment."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
from models.model import NAGL


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lots, images = {}, set()
    for row in rows:
        role, lot = row["split"].strip(), row["lot"].strip()
        if role not in ("ref", "train", "val", "test") or not lot or not row["equipment"].strip():
            raise ValueError(f"Invalid split, equipment or root lot: {row}")
        if lots.setdefault(lot, role) != role:
            raise ValueError(f"Lot crosses splits: {lot}")
        row.update(split=role, lot=lot, label=int(row["label"]))
        if row["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if row["label"] and role in ("train", "ref") and not row["mask"]:
            raise ValueError(f"Missing defect mask for {row['image']}")
        image = path.parent / row["image"]
        with Image.open(image) as im:
            rgb = im.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if digest in images:
            raise ValueError(f"Duplicate image: {image}")
        images.add(digest)
        row.update(path=image, digest=digest, maskpath=path.parent / row["mask"] if row["mask"] else None)
    return rows


def load_image(row, side):
    with Image.open(row["path"]) as im:
        rgb = im.convert("RGB")
    shape = rgb.size
    w, h = (max(1, round(n * side / max(shape))) for n in shape)
    x, y = (side - w) // 2, (side - h) // 2
    canvas = Image.new("RGB", (side, side), (124, 116, 104))
    canvas.paste(rgb.resize((w, h), Image.Resampling.BICUBIC), (x, y))
    mask = torch.zeros(side, side)
    if row["maskpath"]:
        with Image.open(row["maskpath"]) as m:
            if m.size != shape:
                raise ValueError(f"Mask size mismatch: {row['maskpath']}")
            fg = np.asarray(m.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
        if bool(fg.any()) != bool(row["label"]):
            raise ValueError(f"Mask and label disagree: {row['image']}")
        mask[y:y+h, x:x+w] = torch.from_numpy(fg)
    image = TF.normalize(TF.to_tensor(canvas), [.485,.456,.406], [.229,.224,.225])
    return image, mask, (x,y,x+w,y+h), shape


def references(rows, cfg):
    groups = sorted({r["equipment"] for r in rows}) if cfg.by_equipment else ["*"]
    result, names = {}, {}
    for group in groups:
        refs, paths = [], []
        for label, count in [(0,cfg.n_shot), (1,cfg.a_shot)]:
            pool = sorted((r for r in rows if r["split"]=="ref" and r["label"]==label
                           and (not cfg.by_equipment or r["equipment"]==group)), key=lambda r:r["image"])
            if len(pool) < count:
                raise ValueError(f"Insufficient references for {group}, label {label}")
            chosen = random.Random(f"{cfg.seed}/{group}/{label}").sample(pool,count)
            loaded = [load_image(r,cfg.size) for r in chosen]
            refs.append([torch.stack([v[i] for v in loaded])[None] for i in (0,1)])
            paths.extend(r["image"] for r in chosen)
        result[group], names[group] = refs, paths
    return result,names


def supports_for(refs, row, cfg, device):
    key = row["equipment"] if cfg.by_equipment else "*"
    return [[t.to(device) for t in pair] for pair in refs[key]]


@torch.inference_mode()
def scores(model, cfg, refs, rows, device, out=None):
    model.eval()
    results = []
    for i,row in enumerate(rows):
        im, _, box, shape = load_image(row,cfg.size)
        refs_i = supports_for(refs,row,cfg,device)
        grid = cfg.size//14
        patch = model(cfg,im[None,None].to(device),None,None,*refs_i,mode="test").reshape(grid,grid).cpu()
        x,y,right,bottom = box
        centers = torch.arange(grid)*14+7
        valid = ((y<=centers)&(centers<bottom))[:,None] & ((x<=centers)&(centers<right))[None,:]
        vals = patch[valid]
        if not vals.numel() or not torch.isfinite(vals).all():
            raise ValueError(f"Invalid patch scores: {row['image']}")
        score = vals.topk(max(1,math.ceil(.01*len(vals)))).values.mean().item()
        result = {k:row[k] for k in ("image","lot","equipment","label")}
        result["score"] = score
        if out:
            heat = Image.fromarray(patch.numpy()).resize((cfg.size,cfg.size),Image.Resampling.BILINEAR)
            result["map"] = f"maps/{i:06d}.npy"
            np.save(out/result["map"],np.asarray(heat.crop(box).resize(shape,Image.Resampling.BILINEAR)))
        results.append(result)
    return results


def main(args):
    rows = read_manifest(args.manifest)
    device = torch.device(args.device)
    if args.command == "eval":
        state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        cfg = SimpleNamespace(**state["config"])
        query = [r for r in rows if r["split"]=="test"]
        if not query or any(r["lot"] in state["lots"] or r["digest"] in state["images"] for r in query):
            raise ValueError("Need independent test images and lots")
        model = NAGL(cfg).to(device)
        model.load_state_dict(state["model"])
        (args.out/"maps").mkdir(parents=True,exist_ok=True)
        results = scores(model,cfg,state["refs"],query,device,args.out)
        threshold = state["threshold"]
        for r in results:
            r["prediction"] = int(r["score"]>=threshold)
        with (args.out/"scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=results[0]); writer.writeheader();writer.writerows(results)
        report = []
        for eq in [None]+sorted({r["equipment"] for r in results}):
            sub = [r for r in results if eq is None or r["equipment"]==eq]
            bad = sum(r["label"] for r in sub); good = len(sub)-bad
            fp = sum(r["prediction"] and not r["label"] for r in sub)
            fn = sum(r["label"] and not r["prediction"] for r in sub)
            report.append(dict(equipment=eq,normal=good,abnormal=bad,fp=fp,fn=fn,
                               recall=1-fn/bad if bad else None,overkill_per_1000=1000*fp/good if good else None))
        (args.out/"metrics.json").write_text(json.dumps(dict(threshold=threshold,metrics=report),indent=2))
        return
    cfg = SimpleNamespace(size=args.size,n_shot=args.n_shot,a_shot=args.a_shot,
                          seed=args.seed,by_equipment=args.by_equipment,
                          backbone_name="dinov2_vits14",num_learnable_proxies=25)
    if cfg.size%14 or cfg.size<70 or min(cfg.n_shot,cfg.a_shot)<1:
        raise ValueError("Use a size divisible by 14 and >=70, shot counts >=1")
    refs,names = references(rows,cfg)
    train = [r for r in rows if r["split"]=="train"]
    val = [r for r in rows if r["split"]=="val"]
    if any({r["label"] for r in s}!={0,1} for s in (train,val)):
        raise ValueError("Both train and val require normal and defect images")
    if args.command=="check":
        for r in rows: load_image(r,cfg.size)
        print("Manifest OK", {s:sum(r["split"]==s for r in rows) for s in ("ref","train","val","test")})
        return
    if not 0<args.target_recall<=1 or args.epochs<1:
        raise ValueError("Invalid target recall or epochs")
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"best.pt").exists(): raise FileExistsError("Use a new output directory")
    rng = random.Random(cfg.seed); torch.manual_seed(cfg.seed)
    model = NAGL(cfg).to(device)
    model.vision_encoder.requires_grad_(False);model.mask_downsample.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    best = None
    for epoch in range(args.epochs):
        model.train();model.vision_encoder.eval();rng.shuffle(train)
        for row in train:
            im,mask,*_ = load_image(row,cfg.size)
            *_,li,lp = model(cfg,im[None,None].to(device),mask[None].to(device),
                             torch.tensor([row["label"]],device=device),*supports_for(refs,row,cfg,device))
            loss = li+lp
            if not torch.isfinite(loss): raise ValueError("Non-finite loss")
            opt.zero_grad();loss.backward();opt.step()
        results = scores(model,cfg,refs,val,device)
        positives = sorted((r["score"] for r in results if r["label"]),reverse=True)
        threshold = positives[math.ceil(args.target_recall*len(positives))-1]
        fp = sum(not r["label"] and r["score"]>=threshold for r in results)
        print(f"Epoch {epoch+1}: validation FP={fp}, threshold={threshold:.6f}",flush=True)
        if best is None or fp<best:
            best = fp
            state = dict(config=vars(cfg),model=model.state_dict(),refs=refs,threshold=threshold,
                         lots=sorted({r["lot"] for r in rows if r["split"]!="test"}),
                         images=[r["digest"] for r in rows if r["split"]!="test"])
            torch.save(state,args.out/"best.tmp");(args.out/"best.tmp").replace(args.out/"best.pt")
            (args.out/"validation.json").write_text(json.dumps(dict(epoch=epoch+1,reference_images=names,
                threshold=threshold,fp=fp,target_recall=args.target_recall),indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    for cmd in ("check","train","eval"):
        p=subs.add_parser(cmd)
        p.add_argument("--manifest",type=Path,required=True)
        p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
        if cmd!="eval":
            p.add_argument("--size",type=int,default=224)
            p.add_argument("--n-shot",type=int,default=1)
            p.add_argument("--a-shot",type=int,default=1)
            p.add_argument("--seed",type=int,default=42)
            p.add_argument("--by-equipment",action="store_true")
        if cmd!="check": p.add_argument("--out",type=Path,required=True)
        if cmd=="eval": p.add_argument("--checkpoint",type=Path,required=True)
        if cmd=="train":
            p.add_argument("--epochs",type=int,default=10)
            p.add_argument("--target-recall",type=float,default=.95)
    main(parser.parse_args())
