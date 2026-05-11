# Evaluation Protocol

This repo is benchmark-ready when it can compare tile routing choices without code changes. Smoke runs prove wiring; longer runs produce report numbers.

## Required Comparisons

Tile-label recall:

```powershell
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode random --random-trials 5 --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode prior --prior-features data\tile_features\bitplane_stats_spatial_train.npz --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode heuristic --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-count --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-greedy --top-k-values 4 8 12 16 20
```

YOLO routing smoke:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector all --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector random --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector prior --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector heuristic --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector oracle-greedy --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector scout --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
```

Longer K sweep, written as explicit commands instead of a new experiment framework:

```powershell
foreach ($k in 4,8,12,16,20) {
  python scripts\run_yolo_tiles.py --selector random --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector prior --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector heuristic --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector oracle-greedy --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector scout --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
}
```

## Metrics To Report

Report class-agnostic detector recall, selected tile count, selected area fraction, detector calls, mean latency, p95 latency, and the timing phase breakdown written by `run_yolo_tiles.py`.

The phase `pipeline_ms_excl_gt` is the main latency number. Ground-truth parsing and match/eval timings are reported for measurement transparency, not deployment latency claims.

## Report Table

| Method | Detector calls | Tile budget | Recall | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Full image YOLO | 1 | full image | measured | measured | measured |
| All tiles | 49 | all tiles | measured | measured | measured |
| Random top-K | K | selected | measured | measured | measured |
| Train-split spatial prior | K | selected | measured | measured | measured |
| Content heuristic | K | selected | measured | measured | measured |
| Oracle-greedy | K | selected | upper bound | measured | measured |
| Cached scout | K | selected | measured | measured | measured |
| Live scout | K | selected | measured | measured | measured |

## Interpretation Rules

Cached scout results isolate detector routing quality. Live scout results are the honest end-to-end latency path.

COCO-pretrained `yolov8n.pt` smoke runs are wiring and tradeoff evidence. Final VisDrone mAP requires a VisDrone-compatible detector or fine-tuning and is outside this cleanup pass.
