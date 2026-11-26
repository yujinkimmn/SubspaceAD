"""Check synthetic P0-P5 artifacts and P0 equivalence to the original PaDiM."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--baseline",type=Path,required=True)
    args=p.parse_args()
    root=args.root
    index=read(root/"index.json")
    assert len(index)==4 and all(r["status"]=="ok" for r in index),index
    report=(root/"index.html").read_text(encoding="utf-8")
    assert "Recall–FPR trade-off" in report and "Backbone training diagnostics" in report
    for proto in index:
        name=proto["protocol"]
        out=root/name
        results=read(out/"results.json")
        assert len(results)==6 and all(r["status"]=="ok" for r in results),results
        old=args.baseline/name/"exp0_padim"
        base=out/"p0_pretrained"
        for split in ("val","test"):
            files=list((old/split).glob("*.npy"))
            assert files, "Run both P0 and Exp0 smoke tests with --save-raw-maps"
            assert {f.name for f in files}=={f.name for f in (base/split).glob("*.npy")}
            for f in files:
                np.testing.assert_allclose(np.load(f),np.load(base/split/f.name),rtol=1e-6,atol=1e-6)
        ref=torch.load(base/"model.pt",map_location="cpu",weights_only=True)
        for r in results:
            exp=r["experiment"]
            d=out/exp
            assert (d/"model.pt").is_file() and (d/"operating_points.json").is_file()
            model=torch.load(d/"model.pt",map_location="cpu",weights_only=True)
            assert model["backbone_name"]=="resnet18.a1_in1k"
            assert model["post_processor"]==r["post_processor"]==read(d/"threshold.json")
            assert torch.equal(model["channels"],ref["channels"])
            if exp!="p0_pretrained":
                b=read(d/"backbone_summary.json")
                assert b["weight_max_abs_change"]>0
                assert (b["mixstyle_calls"]>0)==("mixstyle" in exp)
                assert any(not torch.equal(v,ref["backbone"][k]) for k,v in model["backbone"].items())
                assert (d/"backbone.pt").is_file()
                saved=torch.load(d/"backbone.pt",map_location="cpu",weights_only=True)
                assert all(torch.equal(v,model["backbone"][k]) for k,v in saved.items())
            for point in r["operating_points"]["points"]:
                if point["kind"]=="fpr_ceiling":
                    assert point["val"]["fpr"]<=point["target"]+1e-7
            print(name,exp,"OK")
    print("P0 equivalence and P0-P5 backbone/threshold artifacts verified")

if __name__=="__main__":
    main()
