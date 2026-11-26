"""Validate native geometry, channel sampling and frozen unused layers for default P0-P5."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    root=p.parse_args().root/"pooled"
    conf=json.loads((root/"config.json").read_text())
    native=tuple(conf["native_shape"])
    assert conf["upsample_factor"]==4 and conf["padim_size"] is None
    assert conf["max_layer_depth"]==2 and conf["padim_dim"]==40
    assert conf["padim_input_shape"]==[v*4 for v in native]
    results=json.loads((root/"results.json").read_text())
    assert len(results)==6 and all(r["status"]=="ok" for r in results),results
    ref=torch.load(root/"p0_pretrained"/"model.pt",map_location="cpu",weights_only=True)
    for r in results:
        d=root/r["experiment"]
        model=torch.load(d/"model.pt",map_location="cpu",weights_only=True)
        assert model["upsample_factor"]==4 and model["max_layer_depth"]==2 and model["n_features"]==40
        assert tuple(model["feature_shape"])==native
        assert tuple(model["mu"].shape)==(native[0]*native[1],40)
        assert tuple(model["chol"].shape)==(native[0]*native[1],40,40)
        assert int(model["channels"].max())<192 and torch.equal(model["channels"],ref["channels"])
        assert all(torch.equal(v,ref["backbone"][k]) for k,v in model["backbone"].items() if not k.startswith(("layer1.","layer2.")))
        for split in ("val","test"):
            assert all(np.load(f).shape==native for f in (d/split).glob("*.npy"))
        if r["experiment"]!="p0_pretrained":
            summary=json.loads((d/"backbone_summary.json").read_text())
            assert summary["weight_max_abs_change"]>0 and summary["max_layer_depth"]==2
            assert (summary["mixstyle_calls"]>0)==("mixstyle" in r["experiment"])
        print(r["experiment"],"native4x depth2 feature40 OK")

if __name__=="__main__":
    main()
