"""Combine official SubspaceAD per-shot benchmark_result.csv files."""

import csv
from pathlib import Path


def main():
    root = Path("results")
    rows = []
    for shot in (1, 2, 4):
        input_path = root / f"bottle_{shot}shot" / "benchmark_results.csv"
        with input_path.open(newline="", encoding="utf-8") as handle:
            scores = list(csv.DictReader(handle))
        if len(scores) != 1 or scores[0]["Category"] != "bottle":
            raise ValueError(f"Unexpected results in {input_path}")
        rows.append({"shots": shot, **scores[0]})
    output = root / "bottle_fewshot_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
