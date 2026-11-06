
import statistics
import pandas as pd

TRACKS = "outputs/layups_pipeline/tracks.csv"
FPS = 29.98

# hoop
HOOP_SHAPE = (0.5, 0.8)    # height / width of a rim+net box (taller boxes include the backboard)
HOOP_SMOOTH = 0.25       # median the hoop box over +/- this many seconds

# ball
ZONE_U = (-1.5, 1.5)       # only trust balls this close to the hoop sideways...
ZONE_V = (-1.5, 2.5)       # ...and up / down
SPOT_SIZE = 0.1            # grid size (hoop widths) for finding fixed objects
FIXED_SHARE = 0.4          # a spot with a "ball" in more than this share of seconds is a fixed object
MAX_SPEED = 15             # fastest the ball moves (hoop widths per second; shots reach the rim at ~10)
JITTER = 0.3               # extra room (hoop widths) for boxes wobbling between frames
LOST_S = 0.5               # a track not seen for this long ends (a ball showing up later starts a new one)

# attempts
ATTEMPT_V = 0.45           # attempt: ball comes up to about rim height...
ATTEMPT_U = 1.0            # ...this close to the hoop sideways...
ATTEMPT_CONF = 0.2         # ...with at least this confidence
SAME_SHOT_S = 0.75         # rim-height moments closer than this are one attempt


# Helper functions
def frames(seconds):
    return round(seconds * FPS)


def to_hoop_coords(box, hoop):
    hoop_width = hoop["x2"] - hoop["x1"]

    # use the center of each box as its position
    ball_center_x = (box["x1"] + box["x2"]) / 2
    ball_center_y = (box["y1"] + box["y2"]) / 2
    hoop_center_x = (hoop["x1"] + hoop["x2"]) / 2

    # divide by hoop width so distances don't depend on how close the camera is
    u = (ball_center_x - hoop_center_x) / hoop_width
    v = (ball_center_y - hoop["y1"]) / hoop_width
    return u, v


def to_spot(u, v):
    return round(u / SPOT_SIZE), round(v / SPOT_SIZE)

def distance(a, b):
    return ((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2) ** 0.5


# Algorithm functions
def load():
    df = pd.read_csv(TRACKS)
    # map all prompts to the correct kind of object
    prompt_to_kind = {"basketball hoop": "hoop",
                     "basketball": "ball",
                     "orange ball": "ball",
                     "person": "person"}
    detections = {"hoop": {}, "ball": {}, "person": {}}

    for row in df.to_dict("records"):
        kind = prompt_to_kind.get(row["class"])
        # skip if no entry
        if kind is None:
            continue
        box = {"x1": row["x1"],
               "y1": row["y1"],
               "x2": row["x2"],
               "y2": row["y2"],
               "confidence": row["confidence"]}
        # make list via frame and box values
        detections[kind].setdefault(row["frame"], []).append(box)

    # frames are numbered 0 to last_frame, so there are last_frame + 1 of them
    last_frame = df["frame"].max()
    num_frames = last_frame + 1
    return detections, num_frames


def steady_hoop(detections, num_frames):
    # filter for most confident hoop ratio box in each frame
    best_hoop_box = {}
    for frame, boxes in detections["hoop"].items():
        for box in boxes:
            ratio = (box["y2"] - box["y1"]) / (box["x2"] - box["x1"])
            # check if valid ratio (with net, bounding box is taller)
            if not (HOOP_SHAPE[0] <= ratio <= HOOP_SHAPE[1]):
                continue
            # take most confident box if there exist two hoop boxes in the frame
            if frame not in best_hoop_box or box["confidence"] > best_hoop_box[frame]["confidence"]:
                best_hoop_box[frame] = box

    # represent hoop as median across time (converted to frames)
    window_frames = frames(HOOP_SMOOTH)
    hoop = [None] * num_frames

    # get nearby boxes within that window
    for frame in range(num_frames):
        nearby_boxes = [best_hoop_box[f] for f in range(frame - window_frames, frame + window_frames + 1)
                        if f in best_hoop_box]
        if nearby_boxes:
            # take the median of the averages boxes for this frame
            hoop[frame] = {}
            for corner in ("x1", "y1", "x2", "y2"):
                hoop[frame][corner] = statistics.median(b[corner] for b in nearby_boxes)
        elif frame > 0:
            # if no hoop, keep most recent value
            hoop[frame] = hoop[frame - 1]
    return hoop


def find_fixed_spots(detections, hoop, num_frames):
    # record which seconds of the video had a ball at each spot
    seconds_at_spot = {}
    for frame, boxes in detections["ball"].items():
        if hoop[frame] is None:
            continue
        for box in boxes:
            u, v = to_hoop_coords(box, hoop[frame])
            spot = to_spot(u, v)
            second = int(frame / FPS)
            seconds_at_spot.setdefault(spot, set()).add(second)

    # find spots in footage that contain stagnant boxes for backgroudn balls
    video_seconds = int(num_frames / FPS)
    # initialize set bc order doesnt matter
    fixed_spots = set()
    for spot_u, spot_v in seconds_at_spot:
        # 3x3 grid around the original spot
        seconds_near_spot = set()
        for step_u in (-1, 0, 1):
            for step_v in (-1, 0, 1):
                neighbor = (spot_u + step_u, spot_v + step_v)
                # add the neighbor's seconds (skip spots where no "ball" was seen)
                if neighbor in seconds_at_spot:
                    seconds_near_spot.update(seconds_at_spot[neighbor])
        # fixed object is there in more than FIXED_SHARE of seconds
        if len(seconds_near_spot) > FIXED_SHARE * video_seconds:
            fixed_spots.add((spot_u, spot_v))
    return fixed_spots


def ball_candidates(detections, hoop, num_frames):
    fixed = find_fixed_spots(detections, hoop, num_frames)
    candidates = [[] for _ in range(num_frames)]
    for frame, boxes in detections["ball"].items():
        if hoop[frame] is None:
            continue
        for box in boxes:
            u, v = to_hoop_coords(box, hoop[frame])
            if to_spot(u, v) in fixed:
                continue
            candidates[frame].append({"u": u, "v": v, "confidence": box["confidence"]})
    return candidates


def ball_tracks(detections, hoop, num_frames):
    # follow every ball at once
    candidates = ball_candidates(detections, hoop, num_frames)
    tracks = []
    for frame in range(num_frames):
        # only keep following tracks seen recently
        active = [t for t in tracks if frame - t["last_frame"] <= frames(LOST_S)]

        # every (candidate, track) pair where the track's ball could have moved to that candidate
        pairs = []
        for i, ball in enumerate(candidates[frame]):
            for j, track in enumerate(active):
                last_ball = track["points"][track["last_frame"]]
                seconds_since = (frame - track["last_frame"]) / FPS
                if distance(ball, last_ball) <= JITTER + MAX_SPEED * seconds_since:
                    pairs.append((distance(ball, last_ball), i, j))

        # closest pairs first with each candidate and each track max one reference
        used_balls, used_tracks = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_balls or j in used_tracks:
                continue
            active[j]["points"][frame] = candidates[frame][i]
            active[j]["last_frame"] = frame
            used_balls.add(i)
            used_tracks.add(j)

        # anything left over is a new ball or a bad detection
        for i, ball in enumerate(candidates[frame]):
            if i not in used_balls:
                tracks.append({"points": {frame: ball}, "last_frame": frame})
    return tracks


def shot_ball(tracks, num_frames):
    track = [None] * num_frames
    for t in tracks:
        for frame, ball in t["points"].items():
            in_zone = ZONE_U[0] < ball["u"] < ZONE_U[1] and ZONE_V[0] < ball["v"] < ZONE_V[1]
            if in_zone and (track[frame] is None or ball["confidence"] > track[frame]["confidence"]):
                track[frame] = ball
    return track


def find_attempts(track):
    attempts = []
    for frame, ball in enumerate(track):
        if ball is None:
            continue
        # determine at rim height, near the hoop sideways, and high enough confidence
        at_rim_height = ball["v"] < ATTEMPT_V and abs(ball["u"]) < ATTEMPT_U and ball["confidence"] >= ATTEMPT_CONF
        if not at_rim_height:
            continue
        # determine if close enough to consider same shot attempt
        if attempts and frame - attempts[-1][1] <= SAME_SHOT_S * FPS:
            attempts[-1][1] = frame     # same shot, so extend frame suration
        else:
            attempts.append([frame, frame])     # new shot
    return attempts


def leaves_net(track, start, end):
    # TODO: implement
    pass


def net_hid_it(track, start, exit_frame, hidden_frames):
    # TODO: implement
    pass


def net_slowed_it(track, exit_frame):
    # TODO: implement
    pass

def is_make(track, start, end):
    # TODO: implement
    pass


def main():
    detections, num_frames = load()
    hoop = steady_hoop(detections, num_frames)
    tracks = ball_tracks(detections, hoop, num_frames)
    track = shot_ball(tracks, num_frames)
    for start, end in find_attempts(track):
        result = "make" if is_make(track, start, end) else "miss"
        print(f"shot attempt at {start / FPS:.1f}s, {result}")


if __name__ == "__main__":
    main()