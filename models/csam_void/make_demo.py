"""Generate toy data for a local smoke test, not a CSAM benchmark."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42)
    rows=[]
    for eq in range(1,4):
        for split in ("train","val","test"):
            lot=f"E{eq}-{split}"
            for i in range(12):
                label=1 if i<4 else (-1 if split=="train" and i>=10 else 0)
                y,x=np.indices((61,81))
                a=70+eq*18+(3+eq*4)*np.sin(y/2)+rng.normal(0,2,(61,81))
                mask=np.zeros((61,81),np.uint8)
                if label==1 or (label==-1 and i%2==0):
                    cy,cx=rng.integers(5,56),rng.integers(5,76)
                    n=1+i%4
                    a[cy,cx:cx+n]+=rng.uniform(60,110)
                    if label==1: mask[cy,cx:cx+n]=255
                stem=f"{lot}-{i}"
                image=stem+".png"
                Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(out/image)
                m=""
                if label==1:
                    m=stem+"_mask.png"
                    Image.fromarray(mask).save(out/m)
                rows.append([image,m,label,split,str(eq),lot,f"{i//6+1:02d}"])
    with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["image","mask","label","split","equipment","lot","wafer"])
        w.writerows(rows)
if __name__=="__main__":
    main()
