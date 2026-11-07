
import statistics
from pathlib import Path

import pandas as pd

TRACKS = "outputs/layups_pipeline/tracks.csv"
FPS = 29.98

# hoop
HOOP_SHAPE = (0.5, 0.8)      # allowed height-to-width ratio of the hoop box; taller boxes include the backboard
HOOP_SMOOTH_S = 0.25         # median window for the hoop box, s on each side of a frame

# ball
ZONE_U = (-1.5, 1.5)         # allowed sideways range from the hoop center, hoop widths
ZONE_V = (-1.5, 2.5)         # allowed vertical range from the rim, hoop widths (negative = above)
SPOT_SIZE = 0.1              # grid spot size for finding fixed objects, hoop widths
FIXED_SHARE = 0.4            # minimum share of the video's seconds for a spot to count as a fixed object
MAX_SPEED = 15               # maximum ball speed, hoop widths/s (shots reach the rim at about 10)
SPEED_CHANGE = 8             # maximum difference from the track's predicted speed, hoop widths/s
JITTER = 0.3                 # extra distance allowed for box wobble between frames, hoop widths
LOST_S = 0.5                 # maximum time a track can go unseen before it ends, s

# release
HOLD_MARGIN = 0.45           # widening of each person's box for the holding check, fraction of box width
RELEASE_MAX_S = 3            # maximum search time before the attempt for the release, s
HOLD_MIN_S = 0.1             # minimum time in a player's hands to count as held, s
POSSESSION_SMOOTH_S = 0.08   # majority vote window for possession, s on each side of a frame
HOLD_TOP = 0.8               # lowest point the ball can be in a person's box and still count as held, fraction of box height from the top

# attempts
ATTEMPT_V = 0.45             # maximum distance below the rim to count as rim height, hoop widths
ATTEMPT_U = 1.0              # maximum sideways distance from the hoop center, hoop widths
ATTEMPT_CONF = 0.2           # minimum detection confidence for an attempt, 0 to 1
SAME_SHOT_S = 0.75           # maximum gap between rim-height moments in the same attempt, s


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
            # take most confident box if there exist multiple hoop boxes in the frame
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
    # initialize nested list to hold candidate info
    candidates = [[] for _ in range(num_frames)]
    for frame, boxes in detections["ball"].items():
        # if our frame has no hoop, continue loop (bc net + ball overlap loses both detections)
        if hoop[frame] is None:
            continue
        for box in boxes:
            u, v = to_hoop_coords(box, hoop[frame])
            # ignore fixed ball spots
            if to_spot(u, v) in fixed:
                continue
            candidates[frame].append({"u": u, "v": v, "confidence": box["confidence"], 
                                      "x": (box["x1"] + box["x2"]) / 2, "y": (box["y1"] + box["y2"]) / 2})
    return candidates

def predict(track, frame):
    # find positiion of latest ball
    last_ball = track["points"][track["last_frame"]]
    # find time since that latest frame
    seconds_since = (frame - track["last_frame"]) / FPS
    if track["speed"] is None:
        # new track, speed unknown, could be anywhere, return search space around position
        return last_ball, JITTER + MAX_SPEED * seconds_since
    # keep moving at its last speed, give or take SPEED_CHANGE, incorrect label cant jump in position to take over ball track
    speed_u, speed_v = track["speed"]
    expected = {"u": last_ball["u"] + speed_u * seconds_since, "v": last_ball["v"] + speed_v * seconds_since}
    return expected, JITTER + SPEED_CHANGE * seconds_since

def ball_tracks(detections, hoop, num_frames):
    # track all ball candidates
    candidates = ball_candidates(detections, hoop, num_frames)
    tracks = []
    for frame in range(num_frames):
        # only keep following tracks that existed recently (to separate lost vs. noise)
        active = [t for t in tracks if frame - t["last_frame"] <= frames(LOST_S)]

        # every (candidate, track) pair where the track's ball could have moved to that candidate
        pairs = []
        for i, ball in enumerate(candidates[frame]):
            for j, track in enumerate(active):
                expected, reach = predict(track, frame)
                if distance(ball, expected) <= reach:
                    pairs.append((distance(ball, expected), i, j))

        # closest pairs first with each candidate and each track only used once at most
        # this is a greedy approach but prob will work for most cases
        used_balls, used_tracks = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_balls or j in used_tracks:
                continue
            # get speed from the last point to this one, for predicting the next frame
            last_ball = active[j]["points"][active[j]["last_frame"]]
            seconds_since = (frame - active[j]["last_frame"]) / FPS
            active[j]["speed"] = ((candidates[frame][i]["u"] - last_ball["u"]) / seconds_since,
                                  (candidates[frame][i]["v"] - last_ball["v"]) / seconds_since)
            active[j]["points"][frame] = candidates[frame][i]
            active[j]["last_frame"] = frame
            candidates[frame][i]["track"] = active[j]
            used_balls.add(i)
            used_tracks.add(j)

        # anything left over is a new ball or a bad detection
        for i, ball in enumerate(candidates[frame]):
            if i not in used_balls:
                tracks.append({"points": {frame: ball}, "last_frame": frame, "speed": None})
                ball["track"] = tracks[-1]
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


def in_hands(ball, person):
    # ball center inside the person's box with HOLD_MARGIN of padding
    margin = HOLD_MARGIN * (person["x2"] - person["x1"])
    return ((person["x1"] - margin <= ball["x"] <= person["x2"] + margin) and (person["y1"] - margin <= ball["y"] <= person["y2"]))


def find_release(shot_track, start, detections):
    # go back along the shot balls path from the rim to the last frame a player had it
    path = shot_track[start]["track"]["points"]
    release = None      # latest in-hands frame of the current run
    for frame in sorted(path, reverse=True):
        if frame > start:
            continue
        if start - frame > frames(RELEASE_MAX_S):
            return None
        people = detections["person"].get(frame, [])
        if any(in_hands(path[frame], person) for person in people):
            if release is None:
                release = frame
            # held long enough, so valid
            if release - frame >= frames(HOLD_MIN_S):
                return release
        else:
            release = None
    return None


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


def possession(detections, tracks, num_frames):
    """
    this is ai generated code
    """
    in_hand = [False] * num_frames
    for track in tracks:
        for frame, ball in track["points"].items():
            for person in detections["person"].get(frame, []):
                above_feet = ball["y"] <= person["y1"] + HOLD_TOP * (person["y2"] - person["y1"])
                if in_hands(ball, person) and above_feet:
                    in_hand[frame] = True
    window = frames(POSSESSION_SMOOTH_S)
    smoothed = []
    for frame in range(num_frames):
        nearby = in_hand[max(0, frame - window):frame + window + 1]
        smoothed.append(sum(nearby) > len(nearby) / 2)
    return smoothed


def export(detections, hoop, tracks, track, num_frames):
    """
    this is ai generated code
    """
    out_dir = Path(TRACKS).parent

    shot_rows = []
    for number, (start, end) in enumerate(find_attempts(track), start=1):
        release = find_release(track, start, detections)
        exit = leaves_net(track, start, end)
        make = is_make(track, start, end)
        shot_rows.append({
            "shot": number,
            "release_frame": release,                     
            "rim_frame": start,
            "exit_frame": exit[0] if exit else None,     
            "result": {True: "make", False: "miss"}.get(make, ""),
            "fps": FPS,
        })
    shots = pd.DataFrame(shot_rows, columns=["shot", "release_frame", "rim_frame", "exit_frame", "result", "fps"])
    shots = shots.astype({"release_frame": "Int64", "exit_frame": "Int64"})
    shots.to_csv(out_dir / "shots.csv", index=False)

    balls_in_frame = [[] for _ in range(num_frames)]
    for t in tracks:
        for frame, ball in t["points"].items():
            balls_in_frame[frame].append(ball)
    best_ball = [None] * num_frames
    current, last_seen = None, None   
    for frame in range(num_frames):
        if current is not None and frame in current["points"]:
            best_ball[frame] = current["points"][frame]
            last_seen = frame
        elif current is not None and frame - last_seen <= frames(LOST_S):
            continue    
        elif balls_in_frame[frame]:
            best_ball[frame] = max(balls_in_frame[frame], key=lambda b: b["confidence"])
            current, last_seen = best_ball[frame]["track"], frame
    has_ball = possession(detections, tracks, num_frames)

    frame_rows = []
    for frame in range(num_frames):
        ball, h = best_ball[frame], hoop[frame]
        frame_rows.append({
            "frame": frame,
            "ball_x": round(ball["x"]) if ball else None,
            "ball_y": round(ball["y"]) if ball else None,
            "hoop_x1": round(h["x1"]) if h else None,
            "hoop_y1": round(h["y1"]) if h else None,
            "hoop_x2": round(h["x2"]) if h else None,
            "hoop_y2": round(h["y2"]) if h else None,
            "possession": has_ball[frame],
            "fps": FPS,
        })
    whole = ["ball_x", "ball_y", "hoop_x1", "hoop_y1", "hoop_x2", "hoop_y2"]
    pd.DataFrame(frame_rows).astype({c: "Int64" for c in whole}).to_csv(out_dir / "frames.csv", index=False)
    print(f"wrote {out_dir / 'shots.csv'} and {out_dir / 'frames.csv'}")


def main():
    detections, num_frames = load()
    hoop = steady_hoop(detections, num_frames)
    tracks = ball_tracks(detections, hoop, num_frames)
    track = shot_ball(tracks, num_frames)
    for start, end in find_attempts(track):
        result = "make" if is_make(track, start, end) else "miss"
        release = find_release(track, start, detections)
        released = f"released at {release / FPS:.1f}s" if release is not None else "release not found"
        print(f"shot attempt at {start / FPS:.1f}s ({released}), {result}")
    export(detections, hoop, tracks, track, num_frames)


if __name__ == "__main__":
    main()