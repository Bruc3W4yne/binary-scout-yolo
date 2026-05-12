#!/usr/bin/env python3
"""Create final report figures from frozen benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "benchmarks" / "oracle_rank_final"
OUT_DIRS = [
    ROOT / "docs" / "report_finalization" / "figures",
    ROOT / "docs" / "overleaf_project" / "figures",
]


RUNS = [
    ("Full YOLO", "full_yolo.json", "baseline", "o"),
    ("Exhaustive", "exhaustive_all_original.json", "reference", "s"),
    ("Oracle K12", "oracle_greedy_k12.json", "oracle", "^"),
    ("Oracle K16", "oracle_greedy_k16.json", "oracle", "^"),
    ("Oracle K18", "oracle_greedy_k18.json", "oracle", "^"),
    ("Oracle K24", "oracle_greedy_k24.json", "oracle", "^"),
    ("GPU objectness K20", "learned_gpu_objectness_k20.json", "gpu", "D"),
    ("GPU oracle-rank K20", "learned_gpu_oracle_rank_k20.json", "gpu", "D"),
    ("Objectness XNOR K12", "objectness_xnor_k12.json", "xnor_obj", "o"),
    ("Objectness XNOR K16", "objectness_xnor_k16.json", "xnor_obj", "o"),
    ("Objectness XNOR K18", "objectness_xnor_k18.json", "xnor_obj", "o"),
    ("Objectness XNOR K20", "objectness_xnor_k20.json", "xnor_obj", "o"),
    ("Objectness XNOR K24", "objectness_xnor_k24.json", "xnor_obj", "o"),
    ("Oracle-rank XNOR K12", "oracle_rank_xnor_k12.json", "xnor_rank", "o"),
    ("Oracle-rank XNOR K16", "oracle_rank_xnor_k16.json", "xnor_rank", "o"),
    ("Oracle-rank XNOR K18", "oracle_rank_xnor_k18.json", "xnor_rank", "o"),
    ("Oracle-rank XNOR K20", "oracle_rank_xnor_k20.json", "xnor_rank", "o"),
    ("Oracle-rank XNOR K24", "oracle_rank_xnor_k24.json", "xnor_rank", "o"),
]

COLORS = {
    "baseline": "#555555",
    "reference": "#222222",
    "oracle": "#1f77b4",
    "gpu": "#2ca02c",
    "xnor_obj": "#ff7f0e",
    "xnor_rank": "#d62728",
}


def load_run(filename: str) -> dict:
    with (ARTIFACTS / filename).open("r", encoding="utf-8") as f:
        doc = json.load(f)
    summary = doc["summary"]
    return {
        "recall": summary["class_agnostic_recall"],
        "small": summary["small_object_recall"],
        "latency": summary["latency_ms"]["mean"],
        "k": summary["top_k"] or 0,
        "tiles": summary["mean_detector_calls"],
    }


def save_all(fig: plt.Figure, name: str) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=220, bbox_inches="tight")


def plot_recall(metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, json_name, group, marker in RUNS:
        run = load_run(json_name)
        ax.scatter(
            run["latency"],
            run[metric],
            s=58,
            color=COLORS[group],
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            label=group,
            zorder=3,
        )
        if label in {
            "Full YOLO",
            "Exhaustive",
            "Oracle K12",
            "GPU objectness K20",
            "Objectness XNOR K18",
            "Oracle-rank XNOR K18",
            "Objectness XNOR K24",
            "Oracle-rank XNOR K24",
        }:
            ax.annotate(label, (run["latency"], run[metric]), xytext=(5, 4), textcoords="offset points", fontsize=7)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), ["baseline", "exhaustive", "oracle", "GPU scout", "objectness XNOR", "oracle-rank XNOR"], fontsize=8)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs latency on 100 VisDrone val images")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    save_all(fig, filename)
    plt.close(fig)


def plot_xnor_sweep() -> None:
    objectness = []
    oracle_rank = []
    for k in [12, 16, 18, 20, 24]:
        objectness.append((k, load_run(f"objectness_xnor_k{k}.json")))
        oracle_rank.append((k, load_run(f"oracle_rank_xnor_k{k}.json")))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot([k for k, _ in objectness], [r["small"] for _, r in objectness], "-o", label="Objectness XNOR small recall", color=COLORS["xnor_obj"])
    ax1.plot([k for k, _ in oracle_rank], [r["small"] for _, r in oracle_rank], "-o", label="Oracle-rank XNOR small recall", color=COLORS["xnor_rank"])
    ax1.set_xlabel("Selected tiles K")
    ax1.set_ylabel("Small-object recall")
    ax1.grid(True, color="#dddddd", linewidth=0.7)
    ax2 = ax1.twinx()
    ax2.plot([k for k, _ in objectness], [r["latency"] for _, r in objectness], "--", label="Objectness XNOR latency", color=COLORS["xnor_obj"], alpha=0.7)
    ax2.plot([k for k, _ in oracle_rank], [r["latency"] for _, r in oracle_rank], "--", label="Oracle-rank XNOR latency", color=COLORS["xnor_rank"], alpha=0.7)
    ax2.set_ylabel("Mean latency (ms)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="lower right")
    ax1.set_title("XNOR K-sweep: oracle-rank supervision helps low-K recall")
    save_all(fig, "xnor_k_sweep.png")
    plt.close(fig)


def plot_recovery() -> None:
    full = load_run("full_yolo.json")
    exhaustive = load_run("exhaustive_all_original.json")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, json_name, group, marker in RUNS[2:]:
        run = load_run(json_name)
        small_recovery = 100 * (run["small"] - full["small"]) / (exhaustive["small"] - full["small"])
        latency_pct = 100 * run["latency"] / exhaustive["latency"]
        ax.scatter(latency_pct, small_recovery, s=58, color=COLORS[group], marker=marker, edgecolor="white", linewidth=0.6)
        if label in {"Oracle K12", "Oracle K18", "GPU objectness K20", "Objectness XNOR K18", "Oracle-rank XNOR K18", "Oracle-rank XNOR K24"}:
            ax.annotate(label, (latency_pct, small_recovery), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
    ax.axvline(100, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Percent of exhaustive tiled latency")
    ax.set_ylabel("Percent of exhaustive small-recall gain recovered")
    ax.set_title("Recall recovery vs compute cost")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    save_all(fig, "small_recall_recovery_vs_latency.png")
    plt.close(fig)


def main() -> None:
    plot_recall("recall", "Overall recall", "recall_vs_latency.png")
    plot_recall("small", "Small-object recall", "small_recall_vs_latency.png")
    plot_xnor_sweep()
    plot_recovery()


if __name__ == "__main__":
    main()
