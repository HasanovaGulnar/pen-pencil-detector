# Pen/Pencil Webcam Detector

Fine-tuned YOLOv8n detector for one custom class: pen_pencil. Uses OpenCV for webcam capture, bounding boxes and live FPS display.

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install ultralytics opencv-python

## Dataset

- Source: Roboflow Universe, phatcara/pens-and-pencils, version 1
- License: CC BY 4.0
- Original classes: Blue-pen, Pencil, Red-pen
- Training class: pen_pencil
- Split: 123 train, 12 validation, 6 test images
- Dataset YAML: dataset/roboflow/data.yaml

All original classes were mapped to pen_pencil by merge_classes.py.

## Train

    python train.py

## Run the webcam detector

    python run_webcam.py --conf 0.75

Use --conf 0.50 to lower the confidence threshold. Press Q in the webcam window to exit. The overlay shows detections, confidence and measured FPS.

The trained weights used by the webcam script are:
models/pen_pencil_best.pt

## Live evaluation

See REAL_WORLD_TEST.md for the lighting/background checks and confidence-threshold comparison.

In the tested empty cluttered scene, conf 0.50 produced a false-positive box; conf 0.75 suppressed it. The recorded rates were 18.5 FPS and 20.3 FPS respectively. FPS varies by computer and scene.

## Dataset attribution

Phatcara, “Pens and pencils,” Roboflow Universe, CC BY 4.0:
https://universe.roboflow.com/phatcara/pens-and-pencils

