"""
Mainly AI-generated file.
Draw the stats on the video from shots.csv + frames.csv (written by stat_tracker_alt.py).
No rules here: everything it shows was decided before, so changing the look never changes the stats.

  - scoreboard: shots / makes / FG%. A shot counts at its release (at the rim if the release is unknown),
    a make or miss counts once it's decided
  - SHOT / MAKE / MISS banners
  - hoop box, the ball with a short trail (cyan while it's a shot, orange otherwise), and who has the ball
  - player box: green, yellow while they have the ball

Usage:
    python overlay_from_csv.py data/layups.mov outputs/layups
Writes overlay_csv.mp4 into that folder.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd

TRAIL_S = .75          # ball trail length
MAX_STEP_PX = 150      # trail segments longer than this (pixels) are jumps between objects, not drawn
SHOT_BANNER_S = 0.7    # how long "SHOT" stays up
RESULT_BANNER_S = 1.5  # how long "MAKE" / "MISS" stays up
MISS_AFTER_S = 1     # a miss with no exit frame is shown this long after the rim

# colors are Blue, Green, Red
BALL, HOOP, WHITE, BLACK = (0, 140, 255), (255, 120, 0), (255, 255, 255), (0, 0, 0)
SHOT_BALL = (255, 255, 0)  # the ball while it's a shot: from release (or rim) until make / miss is decided
PLAYER, PLAYER_HAS_BALL = (0, 200, 0), (0, 255, 255)
BANNER_SCALE = 4.0
MAKE, MISS, SHOT, UNKNOWN = (0, 200, 0), (0, 0, 230), (0, 220, 255), (170, 170, 170)


def text(img, words, pos, color, scale=1.2):
    # outline so it reads on any background; thicken with scale so big text stays bold
    thick = max(2, round(2 * scale / 1.2))
    cv2.putText(img, words, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick * 3)
    cv2.putText(img, words, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def banner(img, words, y, color):
    # center horizontally at any banner size
    thick = max(2, round(2 * BANNER_SCALE / 1.2))
    width = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, BANNER_SCALE, thick)[0][0]
    text(img, words, ((img.shape[1] - width) // 2, y), color, BANNER_SCALE)


def shot_events(shots, fps):
    """When each shot is counted and when its result is shown."""
    events = []
    for s in shots.itertuples():
        shot_frame = s.release_frame if pd.notna(s.release_frame) else s.rim_frame
        if pd.notna(s.exit_frame):
            result_frame = s.exit_frame
        else:
            result_frame = s.rim_frame + round(MISS_AFTER_S * fps)
        result = s.result if isinstance(s.result, str) and s.result else "?"
        events.append({"shot_frame": int(shot_frame), "result_frame": int(result_frame), "result": result})
    return events


def main():
    video_path, folder = sys.argv[1], Path(sys.argv[2])
    shots = pd.read_csv(folder / "shots.csv")
    frames = pd.read_csv(folder / "frames.csv").set_index("frame")
    fps = frames["fps"].iloc[0]
    events = shot_events(shots, fps)
    in_shot = lambda f: any(e["shot_frame"] <= f <= e["result_frame"] for e in events)

    video = cv2.VideoCapture(video_path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = folder / "overlay_csv.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame = 0
    while True:
        ok, img = video.read()
        if not ok:
            break
        row = frames.loc[frame] if frame in frames.index else None

        if row is not None and pd.notna(row.hoop_x1):
            cv2.rectangle(img, (int(row.hoop_x1), int(row.hoop_y1)), (int(row.hoop_x2), int(row.hoop_y2)), HOOP, 3)
        # skip on CSVs exported before player columns existed
        if row is not None and "player_x1" in frames.columns and pd.notna(row.player_x1):
            color = PLAYER_HAS_BALL if row.possession else PLAYER
            cv2.rectangle(img, (int(row.player_x1), int(row.player_y1)), (int(row.player_x2), int(row.player_y2)), color, 3)

        # ball and its trail
        trail = frames.loc[max(0, frame - round(TRAIL_S * fps)):frame, ["ball_x", "ball_y"]].dropna()
        points = [(f, int(x), int(y)) for f, x, y in trail.itertuples()]
        for (fa, xa, ya), (fb, xb, yb) in zip(points, points[1:]):
            # don't draw across frames where the ball wasn't seen, or across a jump no ball could make
            if fb - fa <= 2 and abs(xb - xa) + abs(yb - ya) <= MAX_STEP_PX:
                cv2.line(img, (xa, ya), (xb, yb), SHOT_BALL if in_shot(fb) else BALL, 3)
        if row is not None and pd.notna(row.ball_x):
            cv2.circle(img, (int(row.ball_x), int(row.ball_y)), 16, SHOT_BALL if in_shot(frame) else BALL, 3)

        # scoreboard: shots counted from their release, results once decided
        taken = [e for e in events if e["shot_frame"] <= frame]
        decided = [e for e in events if e["result_frame"] <= frame and e["result"] != "?"]
        makes = sum(e["result"] == "make" for e in decided)
        pct = f"{100 * makes / len(decided):.0f}%" if decided else "-"
        text(img, f"Shots {len(taken)}   Makes {makes}   FG {pct}", (30, 60), WHITE)
        if row is not None:
            text(img, "Has ball" if row.possession else "No ball", (30, 115), MAKE if row.possession else UNKNOWN, 1.0)

        # banners
        for e in events:
            if e["shot_frame"] <= frame < e["shot_frame"] + round(SHOT_BANNER_S * fps):
                banner(img, "SHOT", 170, SHOT)
            if e["result_frame"] <= frame < e["result_frame"] + round(RESULT_BANNER_S * fps):
                color = {"make": MAKE, "miss": MISS}.get(e["result"], UNKNOWN)
                banner(img, e["result"].upper(), 320, color)

        writer.write(img)
        frame += 1

    writer.release()
    print(f"wrote {out_path} ({frame} frames)")


if __name__ == "__main__":
    main()
