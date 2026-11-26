import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from comparison_metrics import sample_evidence, write_evidence
from build_comparison_report import (BASE, align_samples, background, build_comparison_report,
                                     load_samples, summarize_pairs)
from visualization import save_panel


class TestPairedComparison(unittest.TestCase):
    def evidence(self):
        # Two separated GT components (1px and 2px); other pixels are background.
        gt=np.zeros((5,6),np.float32)
        gt[0,0]=1
        gt[3,3:5]=1
        row=dict(id="2",x=np.zeros_like(gt),y=gt,label=1,equipment="<E1>",lot="LOT",wafer="007")
        p0=np.zeros_like(gt); p0[0,0]=.9; p0[1,2]=.8; p0[2,0]=.7
        p1=np.zeros_like(gt); p1[3,4]=.9; p1[1,2]=.5  # equality is NOT a prediction
        normal=dict(row,id="3",label=0,y=np.zeros_like(gt),equipment="E2")
        n0=np.zeros_like(gt); n0[0,:3]=.6
        n1=np.zeros_like(gt); n1[0,0]=.2
        return ([sample_evidence(row,p0,True),sample_evidence(normal,n0,True)],
                [sample_evidence(row,p1,True),sample_evidence(normal,n1,False)])

    def test_same_recall_can_hide_lost_gt(self):
        a,b=self.evidence()
        pairs=align_samples({x["id"]:x for x in a},{x["id"]:x for x in reversed(b)})
        s=summarize_pairs(pairs)
        all_regions=s["regions"]["all"]
        self.assertEqual((all_regions["base_hits"],all_regions["candidate_hits"]),(1,1))
        self.assertEqual((all_regions["retained"],all_regions["lost"],all_regions["gained"]),(0,1,1))
        self.assertEqual(all_regions["retention"],0)
        self.assertEqual(all_regions["base_detected_coverage"],1)
        self.assertEqual(all_regions["candidate_detected_coverage"],0)
        self.assertEqual(s["regions"]["1px"]["lost"],1)
        self.assertEqual(s["regions"]["2-4px"]["gained"],1)
        self.assertIsNone(s["regions"]["2-4px"]["retention"])
        self.assertEqual(s["normal"]["lost"],1)
        pa,pb=s["bg_all"]
        self.assertEqual((pa["fp"],pb["fp"],pa["total"]),(5,0,57))
        self.assertEqual(sum(pa["hist"]),57)
        self.assertEqual(background(pairs,1)[0]["fp"],2)
        self.assertEqual(background(pairs,0)[0]["fp"],3)

    def test_alignment_rejects_different_gt_or_ids(self):
        a,b=self.evidence()
        with self.assertRaisesRegex(ValueError,"test IDs differ"):
            align_samples({x["id"]:x for x in a},{"2":b[0]})
        b[0]["fingerprint"]="changed"
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            align_samples({x["id"]:x for x in a},{x["id"]:x for x in b})

    def test_html_and_rebuild_without_npy(self):
        a,b=self.evidence()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            proto=root/"pooled"; proto.mkdir()
            exp="p1_classification"
            (root/"index.json").write_text(json.dumps([dict(protocol="pooled",status="ok")]))
            (proto/"split_audit.json").write_text(json.dumps({"test":[{"id":"2"},{"id":"3"}]}))
            (proto/"results.json").write_text(json.dumps([
                dict(experiment=BASE,status="ok"),
                dict(experiment=exp,status="ok"),
                dict(experiment="p2_classification_mixstyle",status="skipped",error="one equipment")]))
            for name,samples in ((BASE,a),(exp,b)):
                dest=proto/name; (dest/"test").mkdir(parents=True)
                write_evidence(dest/"comparison_stats.json",samples)
                for rec in samples:
                    Image.new("RGB",(12,8)).save(dest/"test"/(rec["id"]+"_panel.png"))
            self.assertEqual(set(load_samples(proto/BASE/"comparison_stats.json")),{"2","3"})
            page=build_comparison_report(root)
            s=page.read_text(encoding="utf-8")
            for phrase in ("P0 대비 개선 효과 비교","P0 vs p1_classification","P0 검출 영역 1개 유실",
                           "신규 검출","background FP 증가","1px","&lt;E1&gt;","data:image/png;base64,","skipped"):
                self.assertIn(phrase,s)
            self.assertNotIn("http://",s)
            self.assertNotIn("https://",s)
            self.assertFalse(list(root.rglob("*.npy")))
            self.assertEqual(build_comparison_report(root).read_text(encoding="utf-8"),s)
            (proto/"split_audit.json").write_text(json.dumps({"test":[{"id":"2"}]}))
            self.assertIn("do not match the protocol test audit",build_comparison_report(root).read_text(encoding="utf-8"))
            (proto/"split_audit.json").write_text(json.dumps({"test":[{"id":"2"},{"id":"3"}]}))
            (proto/exp/"comparison_stats.json").unlink()
            self.assertIn("정확한 paired 비교 불가",build_comparison_report(root).read_text(encoding="utf-8"))

    def test_pixel_exact_display_scale_for_both_orientations(self):
        # Stub only the OpenCV colormap dependency; check the exact RGB pixels it returns.
        colormap=lambda z: np.repeat((np.clip(z,0,1)*255).astype(np.uint8)[...,None],3,axis=2)
        for h,w in ((61,81),(81,61)):
            with tempfile.TemporaryDirectory() as d, patch("visualization.jet_heatmap",side_effect=colormap):
                x=np.zeros((h,w),np.float32); x[10,20]=.9
                norm=x.copy()
                p=Path(d)/"panel.png"
                save_panel(x,norm,.5,p)
                with Image.open(p) as im:
                    raw=np.asarray(im)
                scale=4; pw,ph=w*scale,h*scale
                ex=lambda z: np.repeat(np.repeat(z,scale,axis=0),scale,axis=1)
                self.assertEqual(raw.shape,(ph+96,3*pw+48,3))
                np.testing.assert_array_equal(raw[32:32+ph,12:12+pw,0],ex((x*255).astype(np.uint8)))
                left=12+pw+12
                np.testing.assert_array_equal(raw[32:32+ph,left:left+pw],ex(colormap(norm)))
                left=12+2*(pw+12)
                np.testing.assert_array_equal(raw[32:32+ph,left:left+pw,0],ex((norm>.5)*255))
                save_panel(x,norm,.5,p,scale=2)
                with Image.open(p) as im:
                    self.assertEqual(im.size,(w*6+48,h*2+96))


if __name__=="__main__":
    unittest.main()
