"""Assert synthetic runs produced the required outputs, not real-world accuracy."""
import argparse
import json
from pathlib import Path
from PIL import Image
import csv
import numpy as np
import torch
from run import post_process

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--semi",action="store_true")
    p.add_argument("--require-semi-update",action="store_true")
    a=p.parse_args()
    index=json.loads((a.root/"index.json").read_text())
    assert (a.root/"index.html").is_file()
    consolidated=(a.root/"index.html").read_text(encoding="utf-8")
    assert "CSAM consolidated report" in consolidated
    assert "Per-equipment metrics" in consolidated
    assert "Test image FPR" in consolidated
    assert (a.root/"summary.csv").is_file()
    for p in index:
        assert p["protocol"] in consolidated
    log=(a.root/"run.log").read_text(encoding="utf-8")
    for token in ("[Dataset]", "[Device]", "[Protocol", "[Data]", "[Experiment", "[Epoch", "[Batch", "[Validation]", "[Memory]", "[Aug]", "Report:"):
        assert token in log, token
    assert "BEST UPDATED" in log or (a.semi and "[SKIPPED]" in log)
    if not a.semi:
        for token in ("[PaDiM fit]", "[Final Val]", "[Final Test]", "protocol_ETA_rough="):
            assert token in log, token
    assert len(index)==(1 if a.semi else 4), index
    expected={"exp7_semi"} if a.semi else {
        "exp0_padim","exp1_cnn","exp2_normals","exp3_hard_negative",
        "exp4_copy_paste","exp5_domain_aug","exp6_mixstyle"}
    for row in index:
        directory=a.root/row["protocol"]
        config=json.loads((directory/"config.json").read_text())
        audit=json.loads((directory/"split_audit.json").read_text())
        h,w=config["native_shape"]
        results=json.loads((directory/"results.json").read_text())
        assert {r["experiment"] for r in results}==expected, results
        assert (directory/"report.html").is_file()
        assert (directory/"summary.csv").is_file()
        for r in results:
            if a.semi and not a.require_semi_update and r["status"]=="skipped":
                assert "No confident positive pseudo-labels accepted" in r["error"],r
                print("Semi-supervised skip behavior verified; training with accepted labels NOT verified")
                continue
            assert r["status"]=="ok",r
            d=directory/r["experiment"]
            for f in ("model.pt","threshold.json","test/scores.csv","val/scores.csv"):
                assert (d/f).is_file(),d/f
            # Final calibrated decisions follow anomalib's F1 thresholds, not the former recall-target raw threshold.
            cal=json.loads((d/"threshold.json").read_text())
            assert cal==r["post_processor"]==torch.load(d/"model.pt",map_location="cpu",weights_only=True)["post_processor"]
            assert cal["source"]=="validation" and cal["threshold"]==.5
            for split in ("val","test"):
                with (d/split/"scores.csv").open(newline="",encoding="utf-8") as f:
                    records=list(csv.DictReader(f))
                by_id={rec["id"]:rec for rec in audit[split]}
                assert {rec["id"] for rec in records}==set(by_id)
                assert {p.name for p in (d/split).glob("*.png")}=={rec["id"]+"_panel.png" for rec in records}
                diagnostic_maps=config.get("save_raw_maps",False)
                if diagnostic_maps:
                    raw=np.stack([np.load(d/split/(rec["id"]+".npy")) for rec in records])
                    scores,norm,labels,masks=post_process(raw,cal)
                    if split=="val":
                        assert cal["image_min"]==float(raw.max(axis=(1,2)).min())
                        assert cal["image_max"]==float(raw.max())
                        assert cal["pixel_min"]==float(raw.min()) and cal["pixel_max"]==float(raw.max())
                else:
                    assert not list((d/split).glob("*.npy"))
                for i,rec in enumerate(records):
                    assert all(rec[key]==str(by_id[rec["id"]][key]) for key in ("lot","wafer","equipment","label"))
                    assert int(rec["prediction"])==int(float(rec["score"])>.5)
                    with Image.open(d/split/(rec["id"]+"_panel.png")) as im:
                        panel=np.asarray(im)
                    top=panel.shape[0]-h-64
                    left=12+2*(w+12)
                    assert top>32 and panel.shape[1]>=3*w+48
                    mask=panel[top:top+h,left:left+w,0]
                    assert np.isin(mask,[0,255]).all()
                    if diagnostic_maps:
                        assert float(rec["score"])==float(scores[i]) and int(rec["prediction"])==int(labels[i])
                        np.testing.assert_array_equal(np.load(d/split/(rec["id"]+"_normalized.npy")),norm[i])
                        np.testing.assert_array_equal(mask>0,masks[i])
            if r["experiment"] != "exp0_padim":
                history=json.loads((d/"history.json").read_text())
                assert history["best_epoch"] is not None
                assert all("paste_successes" in e and "best_updated" in e and "mixstyle_calls" in e for e in history["epochs"])
                if a.semi:
                    assert history["pseudo_positive_pixels"] > 0
            assert list((d/"test").glob("*_panel.png"))
            print(row["protocol"],r["experiment"],"OK")
    print("Synthetic output checks passed")
if __name__=="__main__":
    main()
