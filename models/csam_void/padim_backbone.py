"""Task-guided ResNet18 adaptation followed by the original frozen PaDiM fit."""
import copy
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from csam_progress import stage_progress, epoch_note
from run import (LOG, SkipExperiment, domain_aug, elapsed, gpu_memory, loss_fn,
                 padim_input, padim_preprocess, restore_head_logits)

PEXPS = ["p0_pretrained", "p1_classification", "p2_classification_mixstyle",
         "p3_consistency", "p4_consistency_mixstyle", "p5_pixel_mixstyle"]

class CrossMix(nn.Module):
    def __init__(self, p=.5, alpha=.1):
        super().__init__()
        self.p, self.alpha, self.applied = p, alpha, 0

    def forward(self, x, domains):
        if not self.training or self.p == 0 or random.random() >= self.p:
            return x
        partners = [[j for j,d in enumerate(domains) if d != own] for own in domains]
        if not partners or not all(partners):
            return x
        perm = torch.tensor([random.choice(ids) for ids in partners], device=x.device)
        mu = x.mean((2,3),keepdim=True).detach()
        std = (x.var((2,3),unbiased=False,keepdim=True)+1e-6).sqrt().detach()
        lam = torch.distributions.Beta(self.alpha,self.alpha).sample((len(x),1,1,1)).to(x.device)
        self.applied += 1
        return (x-mu)/std*(lam*std+(1-lam)*std[perm])+lam*mu+(1-lam)*mu[perm]

def preprocess(x,size,upsample_factor=None):
    return padim_preprocess(x,size,upsample_factor)

def layers(net,x,mix=None,domains=(),depth=3):
    stem=net.maxpool(net.act1(net.bn1(net.conv1(x))))
    features=[]
    for i in range(1,depth+1):
        if mix is not None:
            stem=mix(stem,domains)
        stem=getattr(net,f"layer{i}")(stem)
        features.append(stem)
    return tuple(features)

class LocalHead(nn.Module):
    def __init__(self,depth=3):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(sum((64,128,256)[:depth]),64,1),nn.ReLU(),nn.Conv2d(64,1,1))

    def forward(self, features, crop, size, native, upsample_factor=None):
        a=features[0]
        z=torch.cat([a]+[F.interpolate(v,size=a.shape[-2:],mode="nearest") for v in features[1:]],1)
        return restore_head_logits(self.head(z),crop,native,size,upsample_factor)

def consistency(student,teacher):
    # Relative MSE at all selected PaDiM layers. Fixed teacher prevents a moving-target collapse.
    return sum((s-t).square().mean()/(t.square().mean().detach()+1e-6)
               for s,t in zip(student,teacher))/len(teacher)

def batch_rows(rows,batch,steps,supervised,seed):
    rng=random.Random(seed)
    # Same class/equipment-balanced sampling policy for matched ablation pairs.
    groups={}
    for r in rows:
        key=(r["label"],r["equipment"]) if supervised else r["equipment"]
        groups.setdefault(key,[]).append(r)
    keys=list(groups)
    for _ in range(steps):
        chosen=[rng.choice(groups[rng.choice(keys)]) for _ in range(batch)]
        if len({r["equipment"] for r in chosen})==1 and len({r["equipment"] for r in rows})>1:
            other=[k for k in keys if groups[k][0]["equipment"]!=chosen[0]["equipment"]]
            chosen[-1]=rng.choice(groups[rng.choice(other)])
        yield chosen

def tensors(rows,args,device,transform=None):
    x=[padim_input(r,transform(r["x"]) if transform else None) for r in rows]
    z,crop=preprocess(x,args.padim_size,args.upsample_factor)
    y=np.stack([r["y"] for r in rows])
    return z.to(device),crop,torch.from_numpy(y[:,None]).to(device)

@torch.no_grad()
def proxy_eval(net,head,teacher,rows,args,device,supervised,batch_done=None):
    scores=[]; labels=[]; losses=[]; inter=predpix=truepix=0
    for start in range(0,len(rows),args.batch):
        rr=rows[start:start+args.batch]
        z,crop,y=tensors(rr,args,device)
        if supervised:
            logits=head(layers(net,z,depth=args.max_layer_depth),crop,args.padim_size,y.shape[-2:],args.upsample_factor)
            scores.extend(logits.flatten(1).amax(1).cpu().tolist())
            labels.extend(r["label"] for r in rr)
            pred=logits>=0
            inter+=int((pred & (y>0)).sum())
            predpix+=int(pred.sum()); truepix+=int(y.sum())
        else:
            changed,_,_=tensors(rr,args,device,lambda x:np.clip(x*1.10+.03,0,1))
            losses.append((len(rr),float(consistency(layers(net,changed,depth=args.max_layer_depth),layers(teacher,z,depth=args.max_layer_depth)))))
        if batch_done is not None:
            batch_done()
    if supervised:
        return {"head_image_auroc":float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                "head_pixel_dice_at_0.5":2*inter/(predpix+truepix) if predpix+truepix else None}
    return {"fixed_photometric_consistency_loss":sum(n*v for n,v in losses)/sum(n for n,v in losses)}

def adapt_backbone(net,exp,parts,args,device,out):
    idx=PEXPS.index(exp)
    if idx==0:
        return
    supervised=idx in (1,2,5)
    use_mix=idx in (2,4,5)
    train=[r for r in parts["train"] if r["label"] in ((0,1) if supervised else (0,))]
    val=[r for r in parts["val"] if r["label"] in ((0,1) if supervised else (0,))]
    if use_mix and len({r["equipment"] for r in train})<2:
        raise SkipExperiment("Cross-equipment MixStyle requires >=2 training equipment domains")
    if not train or not val:
        raise SkipExperiment("Missing eligible backbone training/validation data")
    teacher=copy.deepcopy(net).eval() if not supervised else None
    if teacher is not None:
        teacher.requires_grad_(False)
    # Freeze ImageNet stem and BN running statistics/affine parameters.
    net.requires_grad_(False)
    for i in range(1,args.max_layer_depth+1):
        getattr(net,f"layer{i}").requires_grad_(True)
    for module in net.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.eval().requires_grad_(False)
    net.eval()  # BN fixed; eval mode does NOT disable gradients.
    head=LocalHead(args.max_layer_depth).to(device) if supervised else None
    mix=CrossMix(args.mix_prob,args.mix_alpha).to(device) if use_mix else None
    groups=[{"params":[p for p in net.parameters() if p.requires_grad],"lr":args.backbone_lr}]
    if head is not None: groups.append({"params":head.parameters(),"lr":args.head_lr})
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    steps=max(args.min_steps,math.ceil(len(train)/args.batch))
    prefixes=tuple(f"layer{i}." for i in range(1,args.max_layer_depth+1))
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if k.startswith(prefixes)}
    history=[]; started=time.perf_counter()
    LOG.info("[Backbone] %s supervised=%s cross_mix=%s train=%d val=%d steps=%d",
             exp,supervised,use_mix,len(train),len(val),steps)
    label=exp.split("_",1)[0].upper()
    val_batches=math.ceil(len(val)/args.batch)
    with stage_progress(f"{label} backbone + validation", args.epochs*(steps+val_batches)) as train_bar:
        for epoch in range(args.epochs):
            train_bar.set_description_str(f"{label} backbone E{epoch+1}/{args.epochs}")
            st=time.perf_counter(); losses=[]; tasks=[]; anchors=[]
            before=mix.applied if mix else 0
            if mix is not None: mix.train()
            for step,rr in enumerate(batch_rows(train,args.batch,steps,supervised,args.seed+epoch),1):
                clean,crop,y=tensors(rr,args,device)
                domains=[r["equipment"] for r in rr]
                if supervised:
                    logits=head(layers(net,clean,mix,domains,args.max_layer_depth),crop,args.padim_size,y.shape[-2:],args.upsample_factor)
                    labels=torch.tensor([r["label"] for r in rr],device=device,dtype=clean.dtype)
                    task=F.binary_cross_entropy_with_logits(logits.flatten(1).amax(1),labels)
                    anchor=loss_fn(logits,y,hard=True) if idx==5 else task.new_zeros(())
                    loss=task+args.pixel_weight*anchor
                else:
                    augmented,_,_=tensors(rr,args,device,domain_aug)
                    with torch.no_grad(): target=layers(teacher,clean,depth=args.max_layer_depth)
                    task=consistency(layers(net,augmented,mix,domains,args.max_layer_depth),target)
                    anchor=consistency(layers(net,clean,depth=args.max_layer_depth),target)
                    loss=task+args.anchor_weight*anchor
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite backbone loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach())); tasks.append(float(task.detach())); anchors.append(float(anchor.detach()))
                train_bar.set_postfix(batch=f"{step}/{steps}", loss=f"{losses[-1]:.4f}", refresh=False)
                train_bar.update(1)
                if args.log_every and (step%args.log_every==0 or step==steps):
                    LOG.info("[Backbone batch %d/%d] epoch=%d loss=%.6f | %s",step,steps,epoch+1,np.mean(losses),gpu_memory(device))
            if mix is not None: mix.eval()
            train_bar.set_description_str(f"{label} proxy validation E{epoch+1}/{args.epochs}")
            train_bar.set_postfix(refresh=False)
            diagnostic=proxy_eval(net,head,teacher,val,args,device,supervised,batch_done=lambda:train_bar.update(1))
            dt=time.perf_counter()-st
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),task_loss=float(np.mean(tasks)),
                     anchor_or_pixel_loss=float(np.mean(anchors)),mixstyle_calls=(mix.applied-before if mix else 0),
                     val=diagnostic,seconds=dt)
            history.append(row)
            (out/"history.json").write_text(json.dumps({"epochs":history,"selected_epoch":args.epochs,
                     "selection":"fixed final epoch; validation proxy is diagnostic only"},indent=2),encoding="utf-8")
            LOG.info("[Backbone epoch %d/%d] loss=%.6f task=%.6f aux=%.6f mix_calls=%d val=%s elapsed=%s ETA=%s",
                     epoch+1,args.epochs,row["loss"],row["task_loss"],row["anchor_or_pixel_loss"],row["mixstyle_calls"],
                     diagnostic,elapsed(dt),elapsed((time.perf_counter()-started)/(epoch+1)*(args.epochs-epoch-1)))
            proxy_key="head_image_auroc" if supervised else "fixed_photometric_consistency_loss"
            proxy_value=diagnostic.get(proxy_key)
            proxy_label="proxy_auc" if supervised else "proxy_loss"
            formatted="N/A" if proxy_value is None else f"{proxy_value:.4f}"
            train_bar.set_postfix(loss=f"{row['loss']:.4f}", **{proxy_label:formatted})
            epoch_note("  %s E%d/%d | loss=%.4f | %s=%s",label,epoch+1,args.epochs,row["loss"],proxy_label,formatted)
    if use_mix and not mix.applied:
        raise SkipExperiment("No cross-equipment MixStyle calls occurred; increase training steps or mix probability")
    delta=max(float((net.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in initial.items() if v.is_floating_point())
    if delta<=0:
        raise ValueError("Backbone weights did not change")
    net.eval().requires_grad_(False)
    torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},out/"backbone.pt")
    (out/"backbone_summary.json").write_text(json.dumps(dict(weight_max_abs_change=delta,
        mixstyle_calls=mix.applied if mix else 0,epochs=args.epochs,max_layer_depth=args.max_layer_depth,
        upsample_factor=args.upsample_factor,n_features=args.padim_dim,final_proxy=history[-1]["val"]),indent=2),encoding="utf-8")
    LOG.info("[Backbone] frozen; weight_delta=%.6g mix_calls=%d; Gaussian fit uses real normals, MixStyle OFF",delta,mix.applied if mix else 0)

def score_samples(rows,maps):
    groups={"normal":[],"positive":[],"1px":[],"2-4px":[],"5+px":[]}
    for r,m in zip(rows,maps):
        groups["positive" if r["label"]==1 else "normal"].append(float(m.max()))
        lab,n=ndimage.label(r["y"]>0,structure=np.ones((3,3)))
        for j in range(1,n+1):
            obj=lab==j
            area=obj.sum()
            groups["1px" if area==1 else "2-4px" if area<=4 else "5+px"].append(float(m[obj].max()))
    return {k:np.sort(np.asarray(v,dtype=np.float64)) for k,v in groups.items()}

def rates(groups,threshold):
    def rate(key):
        a=groups[key]
        return float((len(a)-np.searchsorted(a,threshold,side="left"))/len(a)) if len(a) else None
    return dict(fpr=rate("normal"),recall=rate("positive"),
                region_1px=rate("1px"),region_2_4px=rate("2-4px"),region_5px=rate("5+px"))

def operating_points(val_rows,vm,test_rows,tm,primary):
    """All operating thresholds come from validation only, including the ROC-like sweep."""
    v,t=score_samples(val_rows,vm),score_samples(test_rows,tm)
    points=[]
    for q in (.5,.7,.8,.9,.95,.98,1.):
        a=v["positive"]
        th=float(a[len(a)-math.ceil(q*len(a))])
        points.append(dict(kind="image_recall",target=q,threshold=th,val=rates(v,th),test=rates(t,th)))
        for key in ("1px","2-4px"):
            a=v[key]
            if len(a):
                th=float(a[len(a)-math.ceil(q*len(a))])
                points.append(dict(kind=key+"_recall",target=q,threshold=th,val=rates(v,th),test=rates(t,th)))
    for q in (0.,.01,.05,.1,.2,.5,1.):
        a=v["normal"]; allowed=math.floor(q*len(a))
        th=float(np.nextafter(np.float32(a[len(a)-allowed-1]),np.float32(np.inf))) if allowed<len(a) else float(min(a[0],v["positive"][0]))
        points.append(dict(kind="fpr_ceiling",target=q,threshold=th,val=rates(v,th),test=rates(t,th)))
    if primary is not None:
        points.append(dict(kind="primary",target=None,threshold=primary,val=rates(v,primary),test=rates(t,primary)))
    # Plot a dense curve on validation-derived candidate thresholds, never optimize on test.
    all_scores=np.unique(np.concatenate([a for a in v.values() if len(a)]))
    indices=np.unique(np.linspace(0,len(all_scores)-1,min(101,len(all_scores)),dtype=int))
    thresholds=[float(np.nextafter(np.float32(all_scores[-1]),np.float32(np.inf)))]+[float(all_scores[i]) for i in indices[::-1]]
    curve=[dict(threshold=th,val=rates(v,th),test=rates(t,th)) for th in thresholds]
    return {"points":points,"curve":curve,"score_counts":{s:{k:len(a) for k,a in g.items()} for s,g in (("val",v),("test",t))}}
