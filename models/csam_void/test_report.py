import csv
import json
import tempfile
import unittest
from pathlib import Path
from build_report import build_report

class TestStandaloneReport(unittest.TestCase):
    def test_rebuild_without_original_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"pooled"
            p.mkdir()
            index=[dict(protocol="pooled",equipment=None,status="partial"),
                   dict(protocol="holdout_01",equipment="<E1>",status="skipped",error="no test positives"),
                   dict(protocol="holdout_02",equipment="E2",status="ok")]
            (root/"index.json").write_text(json.dumps(index))
            m=dict(recall=.95,fpr=.125,dice=.6,iou=.43,tp=19,fn=1,fp=1,tn=7,
                   regions={"1px":{"recall":.9,"hit":9,"count":10}})
            m["by_equipment"]={"<E1>":dict(m)}
            results=[dict(experiment="exp0_padim",status="ok",threshold=.5,seconds=1,val=m,test=m),
                     dict(experiment="exp7_semi",status="skipped",error="<not enough>")]
            (p/"results.json").write_text(json.dumps(results))
            (p/"selected.json").write_text(json.dumps({"selected_by_validation":"exp0_padim"}))
            target=build_report(root)
            s=target.read_text(encoding="utf-8")
            self.assertIn("12.50%",s)
            self.assertIn("exp0_padim ★",s)
            self.assertIn("&lt;E1&gt;",s)
            self.assertIn("&lt;not enough&gt;",s)
            self.assertIn("results.json missing",s)
            self.assertIn("no test positives",s)
            self.assertIn("처음 보는 분을 위한 결과 읽는 법",s)
            self.assertIn("지표 사전",s)
            self.assertNotIn("id='padim-guide'",s)
            self.assertNotIn("src=",s)
            self.assertNotIn("pooled/report.html",s)
            self.assertFalse((p/"report.html").exists())
            with (root/"summary.csv").open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),7)
            self.assertEqual(rows[0]["scope"],"overall")
            self.assertEqual(rows[1]["equipment"],"<E1>")
            self.assertEqual(rows[0]["1px_hit"],"9")
            self.assertEqual(rows[-1]["status"],"missing")
            self.assertEqual(build_report(root),target)
            self.assertEqual(target.read_text(encoding="utf-8"),s)
            (root/"comparison.html").write_text("<html>comparison</html>",encoding="utf-8")
            self.assertIn("href='comparison.html'",build_report(root).read_text(encoding="utf-8"))

    def test_padim_guide_and_matched_validation_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"pooled"
            p.mkdir()
            index=[dict(protocol="pooled",equipment=None,status="partial")]
            (root/"index.json").write_text(json.dumps(index))

            def metric(fpr,recall,small):
                return dict(fpr=fpr,recall=recall,worst_equipment_fpr=fpr,
                            regions={"1px":{"recall":small}})

            base=metric(.10,.95,.80)
            improved=metric(.05,.95,.90)
            # Test has deliberately opposite changes; matched deltas must use validation only.
            results=[
                dict(experiment="p0_pretrained",status="ok",val=base,test=improved),
                dict(experiment="p1_classification",status="ok",val=improved,test=base),
                dict(experiment="p2_classification_mixstyle",status="skipped",error="one equipment"),
                dict(experiment="p3_consistency",status="ok",val=base,test=base),
                dict(experiment="p4_consistency_mixstyle",status="failed",error="failed"),
            ]
            (p/"results.json").write_text(json.dumps(results))
            (p/"selected.json").write_text(json.dumps({"selected_by_validation":"p1_classification"}))
            (p/"config.json").write_text(json.dumps({"padim_input_shape":[244,324],"upsample_factor":4,
                "max_layer_depth":2,"padim_dim":40,"padim_map_restore":"bilinear_input_then_native"}))
            s=build_report(root).read_text(encoding="utf-8")
            self.assertIn("id='padim-guide'",s)
            self.assertIn("P0–P5: 무엇이 다른가?",s)
            self.assertIn("anomaly map restoration: bilinear_input_then_native",s)
            self.assertIn("실험별 학습 효과 비교 (validation)",s)
            self.assertIn("P0 → P1",s)
            self.assertIn("P1 → P2",s)
            self.assertIn("-5.00 pp",s)
            self.assertIn("+10.00 pp",s)
            self.assertIn("+0.00 pp",s)
            self.assertIn("—",s)
            self.assertNotIn("+5.00 pp",s)
            self.assertIn("p1_classification ★",s)
            self.assertNotIn("src=",s)
            self.assertNotIn("https://",s)
if __name__=="__main__":
    unittest.main()

