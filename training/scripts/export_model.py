"""Export a trained PaddleOCR recognition model for inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paddleocr.tools.export_model import main as paddle_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PaddleOCR recognition checkpoint to inference format.")
    parser.add_argument("--config", type=Path, required=True, help="Training configuration used for the run")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory holding the checkpoint to export")
    parser.add_argument("--save-dir", type=Path, required=True, help="Destination directory for inference model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.model_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "export_model.py",
        "--config",
        str(args.config.resolve()),
        "--model_dir",
        str(args.model_dir.resolve()),
        "--save_dir",
        str(args.save_dir.resolve()),
    ]
    paddle_export()


if __name__ == "__main__":
    main()
