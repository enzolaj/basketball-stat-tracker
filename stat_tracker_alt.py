"""
Turns tracks.csv into shots.csv + frames.csv for scripts/overlay_from_csv.py
"""

import statistics
from pathlib import Path

import pandas as pd

TRACKS = "outputs/layups_pipeline/tracks.csv"
FPS = 29.98

# hoop: reject backboard-sized boxes, median out detector wobble
HOOP_SHAPE = (0.5, 0.8)     # allowed height-to-width ratio of the hoop box; taller boxes include the backboard
HOOP_SMOOTH_S = 0.25        # median window for the hoop box, s on each side of a frame

# balls and paths: gate matches by speed so junk can't steal the ball
HOOP_ZONE_U = (-1.5, 1.5)   # allowed sideways range from the hoop center, hoop widths
HOOP_ZONE_V = (-1.5, 2.5)   # allowed vertical range from the rim, hoop widths (negative = above)
FIXED_SPOT_SIZE = 0.1       # grid spot size for finding fixed objects, hoop widths
FIXED_SHARE = 0.4           # minimum share of the video's seconds for a spot to count as a fixed object
BALL_MAX_SPEED = 15         # maximum ball speed, hoop widths/s (shots reach the rim at about 10)
MAX_SPEED_CHANGE = 8        # maximum difference from the path's predicted speed, hoop widths/s
BOX_JITTER = 0.3            # extra distance allowed for box wobble between frames, hoop widths
PATH_LOST_S = 0.5           # maximum time a path can go unseen before it ends, s

# shot attempts: trigger at rim height, merge rim bounces into one attempt
RIM_V = 0.45                # maximum distance below the rim to count as rim height, hoop widths
RIM_U = 1.0                 # maximum sideways distance from the hoop center, hoop widths
RIM_CONF = 0.2              # minimum detection confidence for an attempt, 0 to 1
SAME_SHOT_S = 0.75          # maximum gap between rim-height moments in the same attempt, s

# release: walk back from the rim to the last real hold
HOLD_MARGIN = 0.45          # widening of each person's box for the holding check, fraction of box width
RELEASE_LOOKBACK_S = 3      # maximum search time before the rim for the release, s
HOLD_MIN_S = 0.1            # minimum time in a player's hands to count as held, s

# possession: skip balls at the feet, vote out flicker
POSSESSION_SMOOTH_S = 0.08  # majority vote window for possession, s on each side of a frame
HOLD_LOWEST = 0.8           # lowest point the ball can be in a person's box and still count as held, fraction of box height

# overlay: keep the drawn ball steady when a hand covers it
DRAW_SWITCH_CONF = 0.5      # minimum confidence to jump the drawn ball to a far-away ball, 0 to 1


# Helper Functions
def to_frames(seconds):
    return round(seconds * FPS)


# Class Definitions
class Box:
    # one detection (ball, hoop or person) in pixels; also used for the steady hoop
    def __init__(self, x1, y1, x2, y2, confidence=1.0):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2

    def coords_of(self, box):
        # (u, v) of box's center relative to this hoop: u sideways from the rim center, v down from the rim;
        # scale by hoop width so thresholds hold at any camera distance
        u = (box.center_x - self.center_x) / self.width
        v = (box.center_y - self.y1) / self.width
        return u, v

    def holds(self, ball):
        # widen by HOLD_MARGIN to catch a ball held out past the body
        margin = HOLD_MARGIN * self.width
        return self.x1 - margin <= ball.x <= self.x2 + margin and self.y1 - margin <= ball.y <= self.y2


class Ball:
    # one ball candidate in one frame
    def __init__(self, frame, box, hoop):
        self.frame = frame
        self.u, self.v = hoop.coords_of(box)       # hoop widths, for tracking and the shot rules
        self.x, self.y = box.center_x, box.center_y  # pixels, for comparing with person boxes
        self.confidence = box.confidence
        self.path = None                             # the BallPath it ends up on (set by BallPath.add)

    @property
    def spot(self):
        # bucket into grid squares to find fixed objects
        return round(self.u / FIXED_SPOT_SIZE), round(self.v / FIXED_SPOT_SIZE)

    @property
    def near_hoop(self):
        return HOOP_ZONE_U[0] < self.u < HOOP_ZONE_U[1] and HOOP_ZONE_V[0] < self.v < HOOP_ZONE_V[1]

    @property
    def at_rim_height(self):
        # require rim height, near the hoop sideways, and a fairly sure detection
        return self.v < RIM_V and abs(self.u) < RIM_U and self.confidence >= RIM_CONF

    def distance_to(self, u, v):
        return ((self.u - u) ** 2 + (self.v - v) ** 2) ** 0.5


class BallPath:
    # one object followed over time
    def __init__(self, ball):
        self.balls = {}      # frame -> Ball
        self.speed = None    # (u, v) hoop widths per second, None until two points
        self.last = None     # the most recent Ball
        self.add(ball)

    def add(self, ball):
        if self.last is not None:
            seconds = (ball.frame - self.last.frame) / FPS
            self.speed = ((ball.u - self.last.u) / seconds, (ball.v - self.last.v) / seconds)
        self.balls[ball.frame] = ball
        self.last = ball
        ball.path = self

    def seen_recently(self, frame):
        return frame - self.last.frame <= to_frames(PATH_LOST_S)

    def predict(self, frame):
        # expect (u, v) at this frame, plus how far off the ball may land
        seconds = (frame - self.last.frame) / FPS
        if self.speed is None:
            # allow anywhere within BALL_MAX_SPEED until the speed is known
            return self.last.u, self.last.v, BOX_JITTER + BALL_MAX_SPEED * seconds
        # assume its last speed, give or take MAX_SPEED_CHANGE,
        # so still junk stays put instead of jumping onto a moving ball
        speed_u, speed_v = self.speed
        return self.last.u + speed_u * seconds, self.last.v + speed_v * seconds, BOX_JITTER + MAX_SPEED_CHANGE * seconds


class Shot:
    # one attempt and everything decided about it
    def __init__(self, number, rim_frame, rim_end_frame):
        self.number = number
        self.rim_frame = rim_frame      # first frame the ball was at rim height
        self.rim_end_frame = rim_end_frame      # last frame the ball was at rim height
        self.release_frame = None       # None = release unknown
        self.exit_frame = None          # None = never came down through the net
        self.result = None              # "make", "miss", or None = not decided


# Algorithmic Functions
def load_detections():
    df = pd.read_csv(TRACKS)
    # merge synonym prompts into one kind
    prompt_to_kind = {"basketball hoop": "hoop", "basketball": "ball", "orange ball": "ball", "person": "person"}
    detections = {"hoop": {}, "ball": {}, "person": {}}    # kind -> frame -> list of Box
    for row in df.to_dict("records"):
        kind = prompt_to_kind.get(row["class"])
        if kind is None:
            continue
        box = Box(row["x1"], row["y1"], row["x2"], row["y2"], row["confidence"])
        detections[kind].setdefault(row["frame"], []).append(box)

    num_frames = df["frame"].max() + 1
    return detections, num_frames


def steady_hoop(detections, num_frames):
    # pick the most confident rim-shaped box per frame; the net makes it a bit taller than wide
    best = {}
    for frame, boxes in detections["hoop"].items():
        rim_shaped = [box for box in boxes if HOOP_SHAPE[0] <= box.height / box.width <= HOOP_SHAPE[1]]
        if rim_shaped:
            best[frame] = max(rim_shaped, key=lambda box: box.confidence)

    # median over +/- HOOP_SMOOTH_S to cancel wobble; carry the last hoop through gaps
    window = to_frames(HOOP_SMOOTH_S)
    hoop = [None] * num_frames
    for frame in range(num_frames):
        nearby = [best[f] for f in range(frame - window, frame + window + 1) if f in best]
        if nearby:
            hoop[frame] = Box(*(statistics.median(getattr(box, edge) for box in nearby) for edge in ("x1", "y1", "x2", "y2")))
        elif frame > 0:
            hoop[frame] = hoop[frame - 1]
    return hoop


def fixed_spots(all_balls, num_frames):
    # flag spots with a "ball" in over FIXED_SHARE of the video's seconds as fixed objects (rim bracket)
    seconds_at_spot = {}
    for frame_balls in all_balls:
        for ball in frame_balls:
            seconds_at_spot.setdefault(ball.spot, set()).add(int(ball.frame / FPS))

    video_seconds = int(num_frames / FPS)
    fixed = set()
    for spot_u, spot_v in seconds_at_spot:
        # pool the 3x3 grid around the spot so jitter across a spot edge still adds up
        seconds_near = set()
        for step_u in (-1, 0, 1):
            for step_v in (-1, 0, 1):
                seconds_near |= seconds_at_spot.get((spot_u + step_u, spot_v + step_v), set())
        if len(seconds_near) > FIXED_SHARE * video_seconds:
            fixed.add((spot_u, spot_v))
    return fixed


def ball_candidates(detections, hoop, num_frames):
    # wrap ball boxes as Balls once a hoop exists, then drop fixed objects
    all_balls = [[] for _ in range(num_frames)]
    for frame, boxes in detections["ball"].items():
        if hoop[frame] is not None:
            all_balls[frame] = [Ball(frame, box, hoop[frame]) for box in boxes]
    fixed = fixed_spots(all_balls, num_frames)
    return [[ball for ball in frame_balls if ball.spot not in fixed] for frame_balls in all_balls]


def follow_balls(candidates, num_frames):
    # link candidates into paths, all balls at once so a second ball gets its own path
    paths = []
    for frame in range(num_frames):
        active = [path for path in paths if path.seen_recently(frame)]

        # collect (candidate, path) pairs within the path's reach
        pairs = []
        for i, ball in enumerate(candidates[frame]):
            for j, path in enumerate(active):
                expected_u, expected_v, reach = path.predict(frame)
                gap = ball.distance_to(expected_u, expected_v)
                if gap <= reach:
                    pairs.append((gap, i, j))

        # assign closest pairs first, each candidate and path once per frame
        used_balls, used_paths = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_balls or j in used_paths:
                continue
            active[j].add(candidates[frame][i])
            used_balls.add(i)
            used_paths.add(j)

        # start a new path per leftover: a new ball or a bad detection
        for i, ball in enumerate(candidates[frame]):
            if i not in used_balls:
                paths.append(BallPath(ball))
    return paths


def hoop_balls(paths, num_frames):
    # pick the most confident ball near the hoop per frame for the shot rules;
    # leave gaps as None since the net hiding the ball is a make clue
    hoop_ball = [None] * num_frames
    for path in paths:
        for frame, ball in path.balls.items():
            if ball.near_hoop and (hoop_ball[frame] is None or ball.confidence > hoop_ball[frame].confidence):
                hoop_ball[frame] = ball
    return hoop_ball


def find_shots(hoop_ball):
    # group rim-height moments within SAME_SHOT_S into one shot, so rim bounces count once
    shots = []
    for frame, ball in enumerate(hoop_ball):
        if ball is None or not ball.at_rim_height:
            continue
        if shots and frame - shots[-1].rim_end_frame <= SAME_SHOT_S * FPS:
            shots[-1].rim_end_frame = frame
        else:
            shots.append(Shot(len(shots) + 1, frame, frame))
    return shots


def find_release(shot, hoop_ball, detections):
    # walk back along the shot ball's path from the rim to the last frame a player held it;
    # None when the path never reaches a player (e.g. the ball left the top of the frame mid-flight)
    path = hoop_ball[shot.rim_frame].path
    held_until = None      # latest in-hands frame of the current run
    for frame in sorted(path.balls, reverse=True):
        if frame > shot.rim_frame:
            continue
        if shot.rim_frame - frame > to_frames(RELEASE_LOOKBACK_S):
            return None
        if any(person.holds(path.balls[frame]) for person in detections["person"].get(frame, [])):
            if held_until is None:
                held_until = frame
            # require HOLD_MIN_S of holding to rule out the ball just passing a hand
            if held_until - frame >= to_frames(HOLD_MIN_S):
                return held_until
        else:
            held_until = None
    return None


def leaves_net(shot, hoop_ball):
    # TODO: implement
    pass


def net_hid_it(shot, hoop_ball, exit_frame, hidden_frames):
    # TODO: implement
    pass


def net_slowed_it(hoop_ball, exit_frame):
    # TODO: implement
    pass


def is_make(shot, hoop_ball):
    # TODO: implement
    pass


# The following code is majorly AI-generated since its mainly for visuals

def find_possession(detections, paths, num_frames):
    # flag frames with a tracked ball in a player's hands, above the feet to skip dribble bounces
    in_hand = [False] * num_frames
    for path in paths:
        for frame, ball in path.balls.items():
            for person in detections["person"].get(frame, []):
                above_feet = ball.y <= person.y1 + HOLD_LOWEST * person.height
                if person.holds(ball) and above_feet:
                    in_hand[frame] = True

    # majority vote over +/- POSSESSION_SMOOTH_S to smooth out flicker
    window = to_frames(POSSESSION_SMOOTH_S)
    smoothed = []
    for frame in range(num_frames):
        nearby = in_hand[max(0, frame - window):frame + window + 1]
        smoothed.append(sum(nearby) > len(nearby) / 2)
    return smoothed


def ball_to_draw(candidates, num_frames):
    drawn_ball = [None] * num_frames
    last = None     # the last ball drawn
    for frame in range(num_frames):
        if last is not None and frame in last.path.balls:
            drawn_ball[frame] = last.path.balls[frame]
        elif candidates[frame]:
            seconds = (frame - last.frame) / FPS if last is not None else PATH_LOST_S + 1
            reach = BOX_JITTER + BALL_MAX_SPEED * seconds
            nearby = [ball for ball in candidates[frame] if last is not None and ball.distance_to(last.u, last.v) <= reach]
            best = max(candidates[frame], key=lambda ball: ball.confidence)
            if nearby:
                drawn_ball[frame] = min(nearby, key=lambda ball: ball.distance_to(last.u, last.v))
            elif seconds > PATH_LOST_S or best.confidence >= DRAW_SWITCH_CONF:
                drawn_ball[frame] = best
        if drawn_ball[frame] is not None:
            last = drawn_ball[frame]
    return drawn_ball


def export(out_dir, shots, hoop, drawn_ball, has_ball, num_frames):
    # write one row per shot and one row per frame; blank = unknown
    shot_rows = [{"shot": shot.number, "release_frame": shot.release_frame, "rim_frame": shot.rim_frame,
                  "exit_frame": shot.exit_frame, "result": shot.result or "", "fps": FPS} for shot in shots]
    shot_table = pd.DataFrame(shot_rows, columns=["shot", "release_frame", "rim_frame", "exit_frame", "result", "fps"])
    # use Int64 to keep frame numbers whole alongside blanks
    shot_table.astype({"release_frame": "Int64", "exit_frame": "Int64"}).to_csv(out_dir / "shots.csv", index=False)

    frame_rows = []
    for frame in range(num_frames):
        ball, h = drawn_ball[frame], hoop[frame]
        frame_rows.append({
            "frame": frame,
            "ball_x": round(ball.x) if ball else None,
            "ball_y": round(ball.y) if ball else None,
            "hoop_x1": round(h.x1) if h else None,
            "hoop_y1": round(h.y1) if h else None,
            "hoop_x2": round(h.x2) if h else None,
            "hoop_y2": round(h.y2) if h else None,
            "possession": has_ball[frame],
            "fps": FPS,
        })
    whole = ["ball_x", "ball_y", "hoop_x1", "hoop_y1", "hoop_x2", "hoop_y2"]
    pd.DataFrame(frame_rows).astype({c: "Int64" for c in whole}).to_csv(out_dir / "frames.csv", index=False)
    print(f"wrote {out_dir / 'shots.csv'} and {out_dir / 'frames.csv'}")


def main():
    detections, num_frames = load_detections()
    hoop = steady_hoop(detections, num_frames)
    candidates = ball_candidates(detections, hoop, num_frames)
    paths = follow_balls(candidates, num_frames)
    hoop_ball = hoop_balls(paths, num_frames)

    shots = find_shots(hoop_ball)
    for shot in shots:
        shot.release_frame = find_release(shot, hoop_ball, detections)
        net_exit = leaves_net(shot, hoop_ball)
        shot.exit_frame = net_exit[0] if net_exit else None
        make = is_make(shot, hoop_ball)
        shot.result = {True: "make", False: "miss"}.get(make)

        released = f"released at {shot.release_frame / FPS:.1f}s" if shot.release_frame is not None else "release not found"
        print(f"shot attempt at {shot.rim_frame / FPS:.1f}s ({released}), {shot.result or '?'}")

    has_ball = find_possession(detections, paths, num_frames)
    drawn_ball = ball_to_draw(candidates, num_frames)
    export(Path(TRACKS).parent, shots, hoop, drawn_ball, has_ball, num_frames)


if __name__ == "__main__":
    main()
