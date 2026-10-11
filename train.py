from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    data_file = Path("dataset/roboflow/data.yaml")
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset YAML tapilmadi: {data_file}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_file),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        patience=10,
        device=device,
        project="runs",
        name="pen_pencil",
        exist_ok=True,
    )

    print("\nTraining bitdi.")
    print("Best model: runs/pen_pencil/weights/best.pt")

if __name__ == "__main__":
    main()
