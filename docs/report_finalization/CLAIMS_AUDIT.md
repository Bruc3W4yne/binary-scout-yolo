# Claims Audit

This audit is the guardrail for final report wording. Any unsupported claim should be removed or rewritten before submission.

| Claim | Status | Evidence | Safe wording |
|---|---|---|---|
| Exhaustive tiling improves small-object recall over full-image YOLO. | Supported | Full YOLO small recall `0.084`; exhaustive tiled YOLO small recall `0.355`. | Exhaustive original-resolution tiling materially improves small-object recall, but costs 49 detector inputs and `107.2 ms` mean latency. |
| Few high-value tiles can recover most of exhaustive tiling's recall. | Supported as oracle upper bound | Oracle K12 reaches recall `0.363`, small recall `0.329`, latency `32.9 ms`; exhaustive is `0.399` / `0.355` / `107.2 ms`. | Oracle routing shows the tile set has strong redundancy and that selective tiling is worth learning. |
| The implemented binary/XNOR scout solves the original speed/recall target. | Unsupported removed | Target was K18 recall `>=0.355`, small `>=0.318`, latency `<=60.1 ms`; oracle-rank XNOR K18 reached `0.344` / `0.309` / `63.8 ms`. | The binary/XNOR scout is implemented and benchmarked, but final detector-level results are partial validation rather than full success. |
| Oracle-rank training improved low-K binary/XNOR ranking. | Partially supported | Scout-only object recall improved at K12 `0.767 -> 0.833`, K16 `0.899 -> 0.922`, K18 `0.942 -> 0.946`; detector K12 improved `0.299/0.259 -> 0.316/0.285`. | Oracle-rank supervision moved useful tiles earlier, especially at K12, but detector-level K18 still missed the target. |
| XNOR K24 recovers much of exhaustive small-object recall. | Supported with caveat | Oracle-rank XNOR K24 small recall `0.320`; exhaustive small recall `0.355`; latency `75.8 ms` versus `107.2 ms`. | XNOR K24 recovers about 90% of exhaustive small-object recall, but uses 24 tiles and is not the low-K headline. |
| Learned GPU heatmap is the deployment target. | Unsupported removed | GPU scout is a useful reference but not the edge-motivated CPU XNOR path. | Learned GPU heatmap is a ranking/speed reference on the RTX 4090, not the claimed edge deployment route. |
| The project proves real drone/Jetson deployment. | Unsupported removed | No Jetson or onboard drone benchmark was run. | The work is edge-motivated and reports desktop GPU/CPU timings; actual edge hardware validation remains future work. |
| The project is state of the art or beats YOLO overall. | Unsupported removed | YOLO full-image is faster, exhaustive tiling has higher recall, and no official VisDrone mAP is reported. | The contribution is a measured recall/latency tradeoff pipeline, not state-of-the-art detection accuracy. |
| The custom detector loss was implemented and improved performance. | Unsupported removed | The final pass kept YOLO weights and detector loss fixed. | The custom loss from the presentation is discussed as future work; this implementation isolates tile routing. |

Final report-facing text should have no unsupported claims left. The unsupported rows above are retained here only as explicit "do not say this" checks.
