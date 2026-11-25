import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

p=argparse.ArgumentParser(description="Generate a small equipment-shift demo")
p.add_argument("--out",type=Path,required=True)
out=p.parse_args().out
out.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(42)
with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["image","mask","label","split","equipment","lot"])
    for split in ("ref","train","val","test"):
        for eq in ("A","B"):
            for label in (0,1):
                a=rng.normal(100,3,(81,61));stem=f"{split}_{eq}_{label}"
                if eq=="B": a[:,::5]+=12
                mask=""
                if label:
                    x,y=rng.integers(9,37,size=2); m=np.zeros(a.shape,np.uint8)
                    m[y:y+8,x:x+8]=255;a[m>0]+=65;mask=stem+"_mask.png";Image.fromarray(m).save(out/mask)
                Image.fromarray(a.clip(0,255).astype(np.uint8)).save(out/(stem+".png"))
                w.writerow([stem+".png",mask,label,split,eq,f"{split}_{eq}"])
