"""
Download VisDrone2019-DET train/val zip files.

The split names and layout match the official VisDrone DET release. The default
URLs use Ultralytics' GitHub release mirror because it is script-friendly.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

URLS = {
    "train": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP  existing zip: {dest}")
        return

    print(f"GET   {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r      {done / total:6.1%}", end="")
        if total:
            print()


def extract(zip_path: Path, data_dir: Path) -> None:
    split_root = data_dir / zip_path.stem
    if (split_root / "images").exists() and (split_root / "annotations").exists():
        print(f"SKIP  extracted split: {split_root}")
        return

    print(f"UNZIP {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--splits", nargs="+", choices=sorted(URLS), default=["train", "val"])
    parser.add_argument("--keep-zips", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_dir = args.data_dir / "_downloads"

    try:
        for split in args.splits:
            zip_path = zip_dir / Path(URLS[split]).name
            download(URLS[split], zip_path)
            extract(zip_path, args.data_dir)
            if not args.keep_zips:
                zip_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== download_visdrone.py ===")
    for split in args.splits:
        print(f"PASS  {args.data_dir / Path(URLS[split]).stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
