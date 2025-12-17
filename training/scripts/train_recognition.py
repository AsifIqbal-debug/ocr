"""Wrapper around PaddleOCR training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paddleocr.tools.program import main as paddle_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR recognition model on NID dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to PaddleOCR YAML config")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/nid_rec"), help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, help="Override epoch count from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # PaddleOCR reads configuration from sys.argv.
    cli_args = [
        "train.py",
        "--config",
        str(config_path),
        "--save_model_dir",
        str(args.save_dir),
    ]
    if args.epochs is not None:
        cli_args.extend(["--epoch_num", str(args.epochs)])

    sys.argv = cli_args
    paddle_train()


if __name__ == "__main__":
    main()
