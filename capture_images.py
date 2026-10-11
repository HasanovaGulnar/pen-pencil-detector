from pathlib import Path
import cv2

save_dir = Path("dataset/images/raw")
save_dir.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Webcam did not open. Check camera permissions or try index 1.")

count = len(list(save_dir.glob("*.jpg")))
print("SPACE = save image | Q = quit")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read a frame from the webcam.")
        break

    preview = frame.copy()
    cv2.putText(
        preview,
        f"Images: {count} | SPACE: save | Q: quit",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Dataset Capture", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        filename = save_dir / f"image_{count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        count += 1
        print(f"Saved {filename} ({count} total)")

camera.release()
cv2.destroyAllWindows()
