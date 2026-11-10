"""Prepare a deterministic, small MVTec AD bottle subset on a CI runner.

Downloads only the bottle category from a public Hugging Face mirror.
The source images and masks are never committed or uploaded as artifacts.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def make_subset(source_root: Path, output_root: Path, seed: int, n_good: int, n_defect: int):
    source = source_root / "bottle"
    destination = output_root / "bottle"
    train = sorted((source / "train" / "good").glob("*.png"))
    good = sorted((source / "test" / "good").glob("*.png"))
    defect_dirs = sorted(p for p in (source / "test").iterdir() if p.is_dir() and p.name != "good")
    if len(train) < 4 or len(good) < n_good or not defect_dirs:
        raise ValueError("Unexpected source dataset layout or insufficient bottle images")

    rng = random.Random(seed)
    candidates = {}
    for folder in defect_dirs:
        paths = sorted(folder.glob("*.png"))
        rng.shuffle(paths)
        candidates[folder.name] = paths

    selected_bad = []
    while len(selected_bad) < n_defect:
        progressed = False
        for defect_type in sorted(candidates):
            if candidates[defect_type] and len(selected_bad) < n_defect:
                selected_bad.append(candidates[defect_type].pop())
                progressed = True
        if not progressed:
            raise ValueError("Not enough anomalous bottle images")

    selected_good = rng.sample(good, n_good)
    for image in train:
        link_or_copy(image, destination / "train" / "good" / image.name)
    for image in selected_good + selected_bad:
        link_or_copy(image, destination / image.relative_to(source))
        if image.parent.name != "good":
            mask = source / "ground_truth" / image.parent.name / f"{image.stem}_mask.png"
            if not mask.exists():
                raise FileNotFoundError(f"Missing ground truth: {mask}")
            link_or_copy(mask, destination / mask.relative_to(source))

    return {
        "seed": seed,
        "train_normal_pool": len(train),
        "test_normal": len(selected_good),
        "test_anomaly": len(selected_bad),
        "test_defect_types": {
            kind: sum(p.parent.name == kind for p in selected_bad)
            for kind in sorted(candidates)
        },
        "test_filenames": [str(p.relative_to(source)) for p in sorted(selected_good + selected_bad)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("datasets/mvtec-bottle-subset"))
    parser.add_argument("--manifest", type=Path, default=Path("results/dataset_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-test", type=int, default=12)
    parser.add_argument("--anomaly-test", type=int, default=12)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    repo = "foersben/mvtec-ad"
    revision = HfApi().repo_info(repo_id=repo, repo_type="dataset").sha
    snapshot_path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["bottle/train/**", "bottle/test/**", "bottle/ground_truth/**"],
        local_dir="datasets/mvtec-source",
        max_workers=4,
    )
    summary = make_subset(
        Path(snapshot_path), args.out, args.seed, args.normal_test, args.anomaly_test
    )
    summary.update({"source": repo, "revision": revision, "license": "CC BY-NC-SA 4.0"})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "test_filenames"}, indent=2))


if __name__ == "__main__":
    main()
