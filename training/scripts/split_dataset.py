"""Create train/validation/test splits from a master label table."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split labelled samples into train/val/test manifests.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to TSV with `image\tlabel` per line")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where split files are stored")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Portion to allocate to training set")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Portion to allocate to validation set")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducible splits")
    return parser.parse_args()


def load_samples(path: Path) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rel_path, label = line.split("\t", maxsplit=1)
            samples.append((rel_path, label))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def main() -> None:
    args = parse_args()
    samples = load_samples(args.labels)

    random.seed(args.seed)
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train_list.txt": samples[:train_end],
        "val_list.txt": samples[train_end:val_end],
        "test_list.txt": samples[val_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, subset in splits.items():
        target = args.output_dir / filename
        with target.open("w", encoding="utf-8") as handle:
            for rel_path, label in subset:
                handle.write(f"{rel_path}\t{label}\n")
        print(f"Wrote {len(subset)} rows -> {target}")


if __name__ == "__main__":
    main()
