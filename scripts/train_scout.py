"""
Train a small linear tile-occupancy scout from extracted tile features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    metadata = json.loads(str(data["metadata_json"].item()))
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    if labels.shape != (features.shape[0],):
        raise ValueError(f"labels shape {labels.shape} does not match features")
    return features, labels, metadata


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = (pred == y).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(acc.item())}


def artifact_stem(feature_mode: str) -> str:
    return feature_mode.replace("-", "_").replace("/", "_")


def build_model(feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if hidden_dim <= 0:
        return torch.nn.Linear(feature_dim, 1)
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, default=ROOT / "data" / "tile_features" / "bitplane_stats_train.npz")
    parser.add_argument("--val-features", type=Path, default=ROOT / "data" / "tile_features" / "bitplane_stats_val.npz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "scout")
    parser.add_argument("--model", default="bitplane-stats")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 uses a linear scout")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        x_train, y_train, metadata = load_features(args.train_features)
        x_val, y_val, _ = load_features(args.val_features)

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std

        device = choose_device(args.device)
        train_ds = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

        model = build_model(x_train.shape[1], args.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        x_val_t = torch.from_numpy(x_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)
        log = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(xb)

            val_metrics = evaluate(model, x_val_t, y_val_t)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / len(train_ds),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            log.append(row)
            print(
                f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f}"
            )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        model_name = artifact_stem(str(metadata.get("feature_mode", args.model)))
        checkpoint_path = args.out_dir / f"scout_{model_name}.pt"
        log_path = args.out_dir / "train_log.json"
        torch.save(
            {
                "feature_mode": metadata.get("feature_mode", args.model),
                "feature_dim": int(x_train.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "mean": torch.from_numpy(mean.astype(np.float32)),
                "std": torch.from_numpy(std.astype(np.float32)),
                "state_dict": model.cpu().state_dict(),
                "train_features": str(args.train_features),
                "val_features": str(args.val_features),
            },
            checkpoint_path,
        )
        log_path.write_text(json.dumps(log, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== train_scout.py ===")
    print(f"PASS  checkpoint={checkpoint_path}")
    print(f"PASS  log={log_path}")
    print(f"PASS  pos_weight={float(pos_weight.item()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
