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
if __name__=="__main__":
    unittest.main()
