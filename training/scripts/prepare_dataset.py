"""Build an LMDB dataset compatible with PaddleOCR from tagged image paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import lmdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LMDB dataset for PaddleOCR training.")
    parser.add_argument("--label-file", required=True, type=Path, help="Path to TSV with `image\tlabel` per line")
    parser.add_argument("--image-root", required=True, type=Path, help="Directory that anchors image paths")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory for LMDB output")
    parser.add_argument("--map-size", type=int, default=2_147_483_648, help="LMDB map size in bytes (default: 2GB)")
    return parser.parse_args()


def load_pairs(label_file: Path, image_root: Path) -> Iterable[Tuple[Path, str]]:
    with label_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rel_path, text = line.split("\t", maxsplit=1)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Malformed label line: {line}") from exc
            yield image_root / rel_path, text


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(args.dest), map_size=args.map_size)
    cache = {}
    index = 0

    with env.begin(write=True) as txn:
        for image_path, label in load_pairs(args.label_file, args.image_root):
            if not image_path.exists():  # pragma: no cover - defensive
                raise FileNotFoundError(f"Missing image referenced in labels: {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image: {image_path}")
            _, encoded = cv2.imencode(".jpg", image)  # compress to keep dataset light
            cache[f"image-{index:09d}".encode()] = encoded.tobytes()
            cache[f"label-{index:09d}".encode()] = label.encode("utf-8")
            index += 1

            if index % 1000 == 0:
                txn.cursor().putmulti(cache)
                cache.clear()
        if cache:
            txn.cursor().putmulti(cache)

    with env.begin(write=True) as txn:
        txn.put("num-samples".encode(), str(index).encode())
        metadata = {"label_file": str(args.label_file), "image_root": str(args.image_root)}
        txn.put("meta".encode(), json.dumps(metadata, ensure_ascii=False).encode("utf-8"))

    env.sync()
    env.close()
    print(f"Created LMDB with {index} samples at {args.dest}")


if __name__ == "__main__":
    main()
