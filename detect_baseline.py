import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class numbers in the COCO dataset that the pretrained model uses.
PERSON = 0
SPORTS_BALL = 32

COLORS = {PERSON: (0, 255, 0), SPORTS_BALL: (0, 140, 255)}
NAMES = {PERSON: "person", SPORTS_BALL: "ball"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--out-dir", default="outputs/baseline")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_dir / "annotated2.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        valid, frame = video.read()
        if not valid:  # no more frames
            break

        # Run the detector on this one frame, keeping only people and balls
        result = model(frame, imgsz=args.imgsz, conf=args.conf, classes=[PERSON, SPORTS_BALL], verbose=False)[0]

        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

            # Draw the box and its label on the frame.
            color = COLORS[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{NAMES[class_id]} {confidence:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        writer.write(frame)

    video.release()
    writer.release()

    print()
    print(f"Results saved in {out_dir}/")


if __name__ == "__main__":
    main()
