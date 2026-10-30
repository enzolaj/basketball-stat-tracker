import argparse
import csv
from pathlib import Path

from ultralytics import YOLOE


def main():
    parser = argparse.ArgumentParser(description="Track the ball in a video with YOLOE + ByteTrack.")
    parser.add_argument("video")
    parser.add_argument("--model", default="models/yoloe-11s-seg.pt")
    parser.add_argument("--prompts", nargs="+", default=["basketball", "orange ball", "basketball hoop", "person"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--out-dir", default="outputs/yoloe_bytetrack")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE(args.model)
    model.set_classes(args.prompts, model.get_text_pe(args.prompts))

    results = model.track(
        args.video,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        agnostic_nms=True,  # several prompts can box the same ball; keep only one
        stream=True,  # one frame at a time instead of holding the whole video in memory
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )

    csv_file = open(out_dir / "tracks.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])

    total_frames = 0
    frames_with_track = 0
    track_ids = set()

    for result in results:
        boxes = result.boxes
        # boxes.id is None when nothing in this frame is being tracked
        if boxes.id is not None:
            frames_with_track += 1
            for track_id, class_id, confidence, box in zip(
                boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            ):
                track_ids.add(track_id)
                x1, y1, x2, y2 = [round(value) for value in box]
                csv_writer.writerow([total_frames, track_id, args.prompts[class_id], round(confidence, 3), x1, y1, x2, y2])

        total_frames += 1
        if total_frames % 300 == 0:
            print(f"  {total_frames} frames done", flush=True)

    csv_file.close()

    print()
    print(f"Frames processed:         {total_frames}")
    print(f"Frames with a tracked box: {frames_with_track} ({100 * frames_with_track / total_frames:.1f}%)")
    print(f"Distinct track IDs:        {len(track_ids)}")
    print(f"Results saved in {out_dir}/")


if __name__ == "__main__":
    main()
