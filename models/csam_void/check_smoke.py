"""Assert synthetic runs produced the required outputs, not real-world accuracy."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--semi",action="store_true")
    a=p.parse_args()
    index=json.loads((a.root/"index.json").read_text())
    assert (a.root/"index.html").is_file()
    assert len(index)==(1 if a.semi else 4), index
    expected={"exp7_semi"} if a.semi else {
        "exp0_padim","exp1_cnn","exp2_normals","exp3_hard_negative",
        "exp4_copy_paste","exp5_domain_aug","exp6_mixstyle"}
    for row in index:
        directory=a.root/row["protocol"]
        results=json.loads((directory/"results.json").read_text())
        assert {r["experiment"] for r in results}==expected, results
        assert (directory/"report.html").is_file()
        assert (directory/"summary.csv").is_file()
        for r in results:
            if a.semi and r["status"]=="skipped":
                assert "No confident positive pseudo-labels accepted" in r["error"],r
                print("Semi-supervised skip behavior verified; training with accepted labels NOT verified")
                continue
            assert r["status"]=="ok",r
            d=directory/r["experiment"]
            for f in ("model.pt","threshold.json","test/scores.csv","val/scores.csv"):
                assert (d/f).is_file(),d/f
            assert r["val"]["recall"]>=.95-1e-7,r
            for k in ("1px","2-4px"):
                assert r["val"]["regions"][k]["recall"]>=.90-1e-7,r
            assert list((d/"test").glob("*.npy"))
            assert list((d/"test").glob("*_mask.png"))
            print(row["protocol"],r["experiment"],"OK")
    print("Synthetic output checks passed")
if __name__=="__main__":
    main()
