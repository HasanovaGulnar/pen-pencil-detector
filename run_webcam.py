import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold: 0.0-1.0")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")

    model_path = "models/pen_pencil_best.pt"
    model = YOLO(model_path)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Webcam did not open. Check camera permissions or try --camera 1.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smooth_box = None
    smooth_conf = 0.0
    missed_frames = 0
    fps_samples = deque(maxlen=30)
    previous_time = time.perf_counter()

    print(f"Model: {model_path} | threshold: {args.conf:.2f}")
    print("Press Q in the video window to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        result = model.predict(
            frame,
            imgsz=640,
            conf=args.conf,
            verbose=False,
        )[0]

        best_detection = None
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().tolist()
            best_index = max(range(len(confidences)), key=confidences.__getitem__)
            best_detection = (
                result.boxes.xyxy[best_index].cpu().tolist(),
                confidences[best_index],
            )

        if best_detection is not None:
            new_box, new_conf = best_detection
            if smooth_box is None:
                smooth_box = new_box
            else:
                alpha = 0.65
                smooth_box = [
                    alpha * new + (1 - alpha) * old
                    for new, old in zip(new_box, smooth_box)
                ]
            smooth_conf = new_conf
            missed_frames = 0
        else:
            missed_frames += 1
            if missed_frames > 2:
                smooth_box = None
                smooth_conf = 0.0

        if smooth_box is not None:
            x1, y1, x2, y2 = map(int, smooth_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"pen_pencil {smooth_conf:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
            )

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time
        if elapsed > 0:
            fps_samples.append(1.0 / elapsed)
        fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | conf threshold: {args.conf:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Pen/Pencil Detector - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


