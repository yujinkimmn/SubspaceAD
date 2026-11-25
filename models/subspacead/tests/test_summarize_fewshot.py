import csv
import tempfile
import unittest
from pathlib import Path

from tools.summarize_fewshot import summarize


class SummarizeFewshotTest(unittest.TestCase):
    def test_nested_official_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            for shot in (1, 2, 4):
                csv_path = root / f"bottle_{shot}shot" / "official_run_name" / "benchmark_results.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["Category", "Seed", "Image AUROC"])
                    writer.writeheader()
                    writer.writerow({"Category": "bottle", "Seed": 42, "Image AUROC": shot / 4})
            out = summarize(root)
            with out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["shots"] for row in rows], ["1", "2", "4"])
            self.assertEqual(rows[-1]["Image AUROC"], "1.0")
