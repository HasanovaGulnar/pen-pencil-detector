# Real-World Webcam Test

Date: 2026-10-11
Model: Fine-tuned YOLOv8n
Weights: `runs/detect/runs/pen_pencil/weights/best.pt`
Dataset: Roboflow Universe, `phatcara/pens-and-pencils`, version 1
Dataset license: CC BY 4.0
Classes merged for training: Blue-pen, Pencil, Red-pen -> `pen_pencil`

## Live results

| Confidence threshold | Observed FPS | Pen visible | Empty cluttered background |
|---|---:|---|---|
| 0.50 | 18.5 | Detected | False-positive box appeared |
| 0.75 | 20.3 | Detected | No box appeared |

The user also reported that the pen was detected in all three tested lighting/background conditions. Exact per-condition FPS values were not recorded.

## Notes

The higher threshold suppressed the false-positive box in the tested empty scene while still detecting the pen. FPS is measured and overlaid by the webcam program; it can vary with the computer and scene.

Source: https://universe.roboflow.com/phatcara/pens-and-pencils

## Dataset split

| Split | Images |
|---|---:|
| Train | 123 |
| Validation | 12 |
| Test | 6 |
| Total | 141 |

## Lighting and background checks

| Condition | Pen detected at conf 0.75 |
|---|---|
| Normal light, simple background | Yes |
| Dim light | Yes |
| Different/cluttered background | Yes |

On an empty cluttered background, conf 0.50 produced a false-positive box; conf 0.75 suppressed it. FPS varied during the live run: 18.5 at conf 0.50 and 20.3 at conf 0.75.
