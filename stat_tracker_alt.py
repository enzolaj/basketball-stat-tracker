"""
From a given CSV (generated from track_yoloe.py), export two CSVs containing shot data for later visualization.
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
NET_EXIT_S = 1.5            # maximum time after the rim for the ball to exit the net, s

# release: walk back from the rim to the last real hold
HOLD_MARGIN = 0.45          # widening of each person's box for the holding check, fraction of box width
RELEASE_LOOKBACK_S = 3      # maximum search time before the rim for the release, s
HOLD_MIN_S = 0.1            # minimum time in a player's hands to count as held, s

# make and miss:
SLOW_WINDOW_S = 0.2
SLOW_MIN_SEEN = 3
SLOW_DROP =  4.5
RIM_EDGE_U = 0.5
NEAR_CENTER_U = 0.3
MIN_HIDDEN_FRAMES = 3


# possession: skip balls at the feet, vote out flicker
POSSESSION_SMOOTH_S = 0.08  # majority vote window for possession, s on each side of a frame
HOLD_LOWEST = 0.8           # lowest point the ball can be in a person's box and still count as held, fraction of box height

# overlay: keep the drawn ball steady when a hand covers it
DRAW_SWITCH_CONF = 0.5      # minimum confidence to jump the drawn ball to a far-away ball, 0 to 1


# Helper Functions
def to_frames(seconds):
    """Converts seconds to a whole number of frames at FPS."""
    return round(seconds * FPS)


# Class Definitions
class Box:
    """One detection box in pixels: a ball, hoop or person, or the smoothed hoop.

    Attributes:
        x1, y1, x2, y2: Corners in pixels, (x1, y1) top left.
        confidence: Detector score, 0 to 1; 1.0 for the smoothed hoop.
    """
    def __init__(self, x1, y1, x2, y2, confidence=1.0):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence

    @property
    def width(self):
        """Width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self):
        """Height in pixels."""
        return self.y2 - self.y1

    @property
    def center_x(self):
        """Center x in pixels."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        """Center y in pixels."""
        return (self.y1 + self.y2) / 2

    def coords_of(self, box):
        """Places another box's center relative to this box as the hoop.

        Scales by hoop width so thresholds hold at any camera distance.

        Args:
            box: Box to place.

        Returns:
            (u, v) in hoop widths: u sideways from the rim center, v down from the rim.
        """
        u = (box.center_x - self.center_x) / self.width
        v = (box.center_y - self.y1) / self.width
        return u, v

    def holds(self, ball):
        """Checks whether this person box holds the ball.

        Widens the sides and top by HOLD_MARGIN to catch a ball held out past the body.

        Args:
            ball: Ball to check.

        Returns:
            True if the ball's center is inside the widened box.
        """
        margin = HOLD_MARGIN * self.width
        return self.x1 - margin <= ball.x <= self.x2 + margin and self.y1 - margin <= ball.y <= self.y2


class Ball:
    """One ball candidate in one frame.

    Args:
        frame: Frame index.
        box: Detector Box for the ball.
        hoop: That frame's hoop Box.

    Attributes:
        frame: Frame index.
        u, v: Center in hoop widths, for tracking and the shot rules.
        x, y: Center in pixels, for comparing with person boxes.
        confidence: Detector score, 0 to 1.
        path: BallPath it ends up on; set by BallPath.add.
    """
    def __init__(self, frame, box, hoop):
        self.frame = frame
        self.u, self.v = hoop.coords_of(box)
        self.x, self.y = box.center_x, box.center_y
        self.confidence = box.confidence
        self.path = None

    @property
    def spot(self):
        """Grid square in FIXED_SPOT_SIZE steps, for finding fixed objects."""
        return round(self.u / FIXED_SPOT_SIZE), round(self.v / FIXED_SPOT_SIZE)

    @property
    def near_hoop(self):
        """True inside the hoop zone, where the ball can be the hoop ball."""
        return HOOP_ZONE_U[0] < self.u < HOOP_ZONE_U[1] and HOOP_ZONE_V[0] < self.v < HOOP_ZONE_V[1]

    @property
    def at_rim_height(self):
        """True at rim height, near the hoop sideways, with a fairly sure detection."""
        return self.v < RIM_V and abs(self.u) < RIM_U and self.confidence >= RIM_CONF

    def distance_to(self, u, v):
        """Distance to (u, v), in hoop widths."""
        return ((self.u - u) ** 2 + (self.v - v) ** 2) ** 0.5


class BallPath:
    """One object followed over time.

    Args:
        ball: First Ball on the path.

    Attributes:
        balls: Frame -> Ball.
        speed: (u, v) in hoop widths per second; None until two points.
        last: Most recent Ball.
    """
    def __init__(self, ball):
        self.balls = {}
        self.speed = None
        self.last = None
        self.add(ball)

    def add(self, ball):
        """Extends the path and updates its speed.

        Args:
            ball: Next Ball, later than the last one.
        """
        if self.last is not None:
            seconds = (ball.frame - self.last.frame) / FPS
            self.speed = ((ball.u - self.last.u) / seconds, (ball.v - self.last.v) / seconds)
        self.balls[ball.frame] = ball
        self.last = ball
        ball.path = self

    def seen_recently(self, frame):
        """Checks whether the path is still alive at a frame.

        Args:
            frame: Current frame.

        Returns:
            True if the last point is within PATH_LOST_S.
        """
        return frame - self.last.frame <= to_frames(PATH_LOST_S)

    def predict(self, frame):
        """Predicts where the path's ball should be at a frame.

        Args:
            frame: Frame to predict.

        Returns:
            (u, v, reach): expected position and the allowed distance from it, in hoop widths.
        """
        seconds = (frame - self.last.frame) / FPS
        if self.speed is None:
            # allow anywhere within BALL_MAX_SPEED until the speed is known
            return self.last.u, self.last.v, BOX_JITTER + BALL_MAX_SPEED * seconds
        # assume its last speed, give or take MAX_SPEED_CHANGE,
        # so still junk stays put instead of jumping onto a moving ball
        speed_u, speed_v = self.speed
        return self.last.u + speed_u * seconds, self.last.v + speed_v * seconds, BOX_JITTER + MAX_SPEED_CHANGE * seconds


class Shot:
    """One shot attempt and everything decided about it.

    Args:
        number: Attempt number, from 1.
        rim_frame: First frame at rim height.
        rim_end_frame: Last frame at rim height.

    Attributes:
        number, rim_frame, rim_end_frame: As above.
        release_frame: Frame the ball left the shooter's hands; None if unknown.
        exit_frame: Frame the ball came down through the net; None if it never did.
        result: "make", "miss", or None if undecided.
    """
    def __init__(self, number, rim_frame, rim_end_frame):
        self.number = number
        self.rim_frame = rim_frame
        self.rim_end_frame = rim_end_frame
        self.release_frame = None
        self.exit_frame = None
        self.result = None


# Algorithmic Functions
def load_detections():
    """Loads TRACKS into boxes by kind and frame.

    Returns:
        A tuple (detections, num_frames):
        detections: Kind ("hoop", "ball", "person") -> frame -> list of Box.
        num_frames: Frame count of the video.
    """
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
    """Builds one steady hoop box per frame.

    Args:
        detections: From load_detections.
        num_frames: Frame count.

    Returns:
        List of Box per frame; None before the first hoop is seen.
    """
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
    """Finds grid spots holding a fixed object the detector calls a ball, like the rim bracket.

    Flags a spot when a "ball" sits there in over FIXED_SHARE of the video's seconds.

    Args:
        all_balls: List of Balls per frame.
        num_frames: Frame count.

    Returns:
        Set of fixed spots, as Ball.spot tuples.
    """
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
    """Wraps ball boxes as Balls once a hoop exists, then drops fixed objects.

    Args:
        detections: From load_detections.
        hoop: From steady_hoop.
        num_frames: Frame count.

    Returns:
        List of Balls per frame.
    """
    all_balls = [[] for _ in range(num_frames)]
    for frame, boxes in detections["ball"].items():
        if hoop[frame] is not None:
            all_balls[frame] = [Ball(frame, box, hoop[frame]) for box in boxes]
    fixed = fixed_spots(all_balls, num_frames)
    return [[ball for ball in frame_balls if ball.spot not in fixed] for frame_balls in all_balls]


def follow_balls(candidates, num_frames):
    """Links ball candidates into paths, frame by frame.

    Tracks all balls at once so a second ball gets its own path.

    Args:
        candidates: From ball_candidates.
        num_frames: Frame count.

    Returns:
        List of every BallPath, including one-frame paths from bad detections.
    """
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
    """Picks the ball the shot rules watch in each frame.

    Leaves gaps as None, since the net hiding the ball is a make clue.

    Args:
        paths: From follow_balls.
        num_frames: Frame count.

    Returns:
        List per frame of the most confident Ball near the hoop, or None.
    """
    hoop_ball = [None] * num_frames
    for path in paths:
        for frame, ball in path.balls.items():
            if ball.near_hoop and (hoop_ball[frame] is None or ball.confidence > hoop_ball[frame].confidence):
                hoop_ball[frame] = ball
    return hoop_ball


def find_shots(hoop_ball):
    """Groups rim-height moments into shot attempts.

    Joins moments within SAME_SHOT_S into one shot, so rim bounces count once.

    Args:
        hoop_ball: From hoop_balls.

    Returns:
        List of Shot in time order, with rim frames set.
    """
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
    """Finds when the ball left the shooter's hands.

    Walks back along the hoop ball's path from the rim, up to RELEASE_LOOKBACK_S,
    to the last frame a player held it.

    Args:
        shot: Shot to check.
        hoop_ball: From hoop_balls.
        detections: From load_detections.

    Returns:
        Release frame, or None when the path never reaches a player.
    """
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
    """Finds where the ball comes down out of the net.

    Args:
        shot: Shot to check.
        hoop_ball: From hoop_balls.

    Returns:
        Tuple starting with the exit frame, or None if the ball never comes down through the net.
    """

    # make sure we are looking at the correct ball
    start_ball = hoop_ball[shot.rim_frame]
    if start_ball is None:
        return None

    path = start_ball.path

    # find last frame where the ball is still in the shot event (above or at rim height)
    previous_frames = [f for f  in path.balls if f <= shot.rim_end_frame]
    previous = path.balls[max(previous_frames)]

    # find the frame with the ball immediately after shot event
    for f in sorted(path.balls):
        if f <= shot.rim_end_frame:
            continue
        if f - shot.rim_end_frame > to_frames(NET_EXIT_S):
            break
        ball = path.balls[f]

        crossed_rim = previous.v <= RIM_V and ball.v > RIM_V
        moving_down = ball.v > previous.v

        if crossed_rim and moving_down:
            return f, ball.u

        # so that eventually the previous should cross the rim
        previous = ball

    return None
# do we have to check that the ball is in the correct x position?
# or is that already checked when the Shot is instantiated?
# check that in is_make


def net_hid_it(shot, hoop_ball, exit_frame, min_hidden_frames):
    """Make clue: the net hid the ball just before it came out.

    Args:
        shot: Shot to check.
        hoop_ball: From hoop_balls.
        exit_frame: From leaves_net.
        hidden_frames: allowable frames the ball can go unseen before the exit.

    Returns:
        True if the net hid the ball.
    """
    if exit_frame is None:
        return False

    # look between last rim_end_frame and exit_frame
    hidden_count = 0
    max_hidden = 0

    path = hoop_ball[shot.rim_frame].path
    for f in range(shot.rim_end_frame + 1, exit_frame):
        if f not in path.balls:
            hidden_count += 1
            max_hidden = max(hidden_count, max_hidden)
        else: hidden_count = 0

    return max_hidden >= min_hidden_frames


def net_slowed_it(hoop_ball, exit_frame):
    """Make clue: the net slowed the ball as it dropped out.

    Args:
        hoop_ball: From hoop_balls.
        exit_frame: From leaves_net.

    Returns:
        True if the ball slowed, None with too few sightings to tell.
    """
    # collect the frames in the window right after the exit where the ball was seen
    window_end = min(len(hoop_ball), exit_frame + to_frames(SLOW_WINDOW_S))
    seen = []
    for f in range(exit_frame, window_end):
        if hoop_ball[f] is not None:
            seen.append(f)

    if len(seen) < SLOW_MIN_SEEN:
        return None

    # average downward speed from the first to the last sighting, hoop widths/s
    first = hoop_ball[seen[0]]
    last = hoop_ball[seen[-1]]
    seconds = (last.frame - first.frame) / FPS
    speed = (last.v - first.v) / seconds
    return 0 <= speed < SLOW_DROP

def is_make(shot, hoop_ball):
    """Decides make or miss from the net clues.

    Args:
        shot: Shot to check.
        hoop_ball: From hoop_balls.

    Returns:
        True for a make, False for a miss, None if undecided.
    """
    net_exit = leaves_net(shot, hoop_ball)
    if net_exit is None:
        return False
    exit_frame, exit_u = net_exit

    # leaves_net no longer rejects misses, so check here that the ball came down inside the rim
    if abs(exit_u) >= RIM_EDGE_U:
        return False

    votes = [abs(exit_u) < NEAR_CENTER_U,
             net_hid_it(shot, hoop_ball, exit_frame, MIN_HIDDEN_FRAMES),
             net_slowed_it(hoop_ball, exit_frame)]

    # need 2 of 3 clues to agree; net_slowed_it can abstain with None
    if votes.count(True) >= 2:
        return True
    if votes.count(False) >= 2:
        return False
    return None

# The following code is majorly AI-generated since its mainly for visuals

def find_possession(detections, paths, num_frames):
    """Flags the frames where a player has the ball.

    Counts a tracked ball in a player's hands, above HOLD_LOWEST of their box to skip dribble bounces.

    Args:
        detections: From load_detections.
        paths: From follow_balls.
        num_frames: Frame count.

    Returns:
        List of bool per frame.
    """
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
    """Picks one ball per frame for the overlay.

    Stays on last frame's path; else takes the ball nearest the last drawn one, since a hand
    over the ball often starts a new path; else jumps to the most confident ball, once it
    reaches DRAW_SWITCH_CONF or the old one is gone for PATH_LOST_S.

    Args:
        candidates: From ball_candidates.
        num_frames: Frame count.

    Returns:
        List per frame of the Ball to draw, or None.
    """
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
    """Writes shots.csv (one row per shot) and frames.csv (one row per frame).

    Leaves cells blank where a value is unknown.

    Args:
        out_dir: Folder to write into, as a Path.
        shots: From find_shots, with results filled in.
        hoop: From steady_hoop.
        drawn_ball: From ball_to_draw.
        has_ball: From find_possession.
        num_frames: Frame count.
    """
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
    """Runs the pipeline on TRACKS and writes the CSVs next to it."""
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
