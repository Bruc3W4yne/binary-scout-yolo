"""
Train the learned heatmap scout used by the live top-K tile router.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from heatmap_scout import (  # noqa: E402
    HEATMAP_SIZE,
    IMAGE_SIZE,
    HeatmapScout,
    heatmap_scout_loss,
    heatmap_target_from_boxes,
    load_checkpoint,
    load_scaled_boxes,
    rgb_to_msb_planes,
    save_checkpoint,
    tile_targets_from_boxes,
)
from preprocess import Box, load_image_size, make_tiles  # noqa: E402


class VisDroneHeatmapDataset(Dataset):
    def __init__(self, root: Path, max_images: int = 0):
        self.root = root
        self.image_dir = root / "images"
        self.annotation_dir = root / "annotations"
        self.stems = [path.stem for path in sorted(self.image_dir.glob("*.jpg"))]
        if max_images:
            self.stems = self.stems[:max_images]
        if not self.stems:
            raise FileNotFoundError(f"no .jpg images found in {self.image_dir}")
        self.tiles = make_tiles(img_size=IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        image_path = self.image_dir / f"{stem}.jpg"
        ann_path = self.annotation_dir / f"{stem}.txt"
        orig_size = load_image_size(image_path)
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS), dtype=np.uint8)
        boxes = load_scaled_boxes(ann_path, orig_size, image_size=IMAGE_SIZE)
        return sample_tensors(rgb, boxes, self.tiles, stem)


class SyntheticHeatmapDataset(Dataset):
    def __init__(self, images: int, seed: int = 42):
        self.images = int(images)
        self.seed = int(seed)
        self.tiles = make_tiles(img_size=IMAGE_SIZE)

    def __len__(self) -> int:
        return self.images

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        rgb = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        boxes = []
        for _ in range(1 + idx % 3):
            w = int(rng.integers(6, 48))
            h = int(rng.integers(6, 48))
            x1 = int(rng.integers(0, IMAGE_SIZE - w))
            y1 = int(rng.integers(0, IMAGE_SIZE - h))
            rgb[y1 : y1 + h, x1 : x1 + w] = np.array([255, 255, 255], dtype=np.uint8)
            boxes.append(Box(float(x1), float(y1), float(x1 + w), float(y1 + h), 0, 1))
        return sample_tensors(rgb, boxes, self.tiles, f"synthetic_{idx:05d}")


def sample_tensors(rgb: np.ndarray, boxes: list[Box], tiles, stem: str) -> dict:
    return {
        "stem": stem,
        "planes": torch.from_numpy(rgb_to_msb_planes(rgb)),
        "heatmap": torch.from_numpy(heatmap_target_from_boxes(boxes)),
        "tile_targets": torch.from_numpy(tile_targets_from_boxes(tiles, boxes)),
    }


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def estimate_pos_weight(dataset: Dataset, sample_limit: int) -> float:
    pos = 0.0
    total = 0.0
    for idx in range(min(len(dataset), sample_limit)):
        heatmap = dataset[idx]["heatmap"]
        pos += float((heatmap > 0).sum().item())
        total += float(heatmap.numel())
    neg = max(0.0, total - pos)
    return float(np.clip(neg / max(pos, 1.0), 10.0, 100.0))


@torch.no_grad()
def evaluate(model: HeatmapScout, loader: DataLoader, device: torch.device, tiles, pos_weight: torch.Tensor) -> dict:
    model.eval()
    total_loss = dense = dice = tile = 0.0
    batches = 0
    for batch in loader:
        planes = batch["planes"].to(device)
        heatmap = batch["heatmap"].to(device)
        tile_targets = batch["tile_targets"].to(device)
        logits = model(planes)
        loss, parts = heatmap_scout_loss(logits, heatmap, tile_targets, tiles, pos_weight)
        total_loss += float(loss.item())
        dense += parts["dense_bce"]
        dice += parts["dice"]
        tile += parts["tile_bce"]
        batches += 1
    model.train()
    denom = max(batches, 1)
    return {
        "loss": total_loss / denom,
        "dense_bce": dense / denom,
        "dice": dice / denom,
        "tile_bce": tile / denom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-train")
    parser.add_argument("--val-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-val")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "heatmap_scout" / "heatmap_scout.pt")
    parser.add_argument("--variant", choices=["float", "ste"], default="float")
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--synthetic-images", type=int, default=0, help="use synthetic data instead of VisDrone")
    parser.add_argument("--pos-weight-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        if args.batch < 1:
            raise ValueError("--batch must be positive")

        device = choose_device(args.device)
        if args.synthetic_images:
            train_ds = SyntheticHeatmapDataset(args.synthetic_images, seed=args.seed)
            val_ds = SyntheticHeatmapDataset(max(1, min(args.synthetic_images, 8)), seed=args.seed + 10000)
            dataset_name = "synthetic"
        else:
            train_ds = VisDroneHeatmapDataset(args.train_root, max_images=args.max_train_images)
            val_ds = VisDroneHeatmapDataset(args.val_root, max_images=args.max_val_images)
            dataset_name = "visdrone"

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        tiles = make_tiles(img_size=IMAGE_SIZE)

        model = HeatmapScout(args.variant)
        if args.init:
            init_model, _ = load_checkpoint(args.init)
            missing, unexpected = model.load_state_dict(init_model.state_dict(), strict=False)
            if unexpected:
                raise ValueError(f"unexpected init keys: {unexpected}")
            print(f"init={args.init} missing={list(missing)}")
        model.to(device)

        pos_weight_value = estimate_pos_weight(train_ds, args.pos_weight_samples)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        log = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = {"loss": 0.0, "dense_bce": 0.0, "dice": 0.0, "tile_bce": 0.0}
            samples = 0
            for batch in train_loader:
                planes = batch["planes"].to(device)
                heatmap = batch["heatmap"].to(device)
                tile_targets = batch["tile_targets"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(planes)
                loss, parts = heatmap_scout_loss(logits, heatmap, tile_targets, tiles, pos_weight)
                loss.backward()
                optimizer.step()
                batch_n = int(planes.shape[0])
                running["loss"] += float(loss.item()) * batch_n
                for key, value in parts.items():
                    running[key] += value * batch_n
                samples += batch_n

            val = evaluate(model, val_loader, device, tiles, pos_weight)
            row = {
                "epoch": epoch,
                "train": {key: value / max(samples, 1) for key, value in running.items()},
                "val": val,
            }
            log.append(row)
            print(
                f"epoch={epoch:03d} train_loss={row['train']['loss']:.4f} "
                f"val_loss={val['loss']:.4f} val_tile_bce={val['tile_bce']:.4f}"
            )

        metadata = {
            "dataset": dataset_name,
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "pos_weight": pos_weight_value,
            "train_log": log,
        }
        save_checkpoint(model, args.out, metadata, ROOT)
        log_path = args.out.with_suffix(".train_log.json")
        log_path.write_text(json.dumps(log, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== train_heatmap_scout.py ===")
    print(f"PASS  checkpoint={args.out}")
    print(f"PASS  log={log_path}")
    print(f"PASS  variant={args.variant} dataset={dataset_name} pos_weight={pos_weight_value:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
