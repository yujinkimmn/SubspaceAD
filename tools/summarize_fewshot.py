"""Combine official SubspaceAD per-shot benchmark_results.csv files."""

import csv
from pathlib import Path


def summarize(root: Path) -> Path:
    rows = []
    for shot in (1, 2, 4):
        matches = sorted((root / f"bottle_{shot}shot").glob("*/benchmark_results.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected one benchmark CSV for {shot}-shot, got {len(matches)}")
        input_path = matches[0]
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
    return output


def main():
    output = summarize(Path("results"))
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
