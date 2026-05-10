# Experimental BNN Detector

This folder preserves the inherited full-BNN detector experiment.

It is not the main project path. The primary pipeline is:

```text
VisDrone image -> tile labels -> scout features -> top-K routing -> YOLO on selected tiles
```

Use this experiment only after the scout + YOLO pipeline is working and measured.
