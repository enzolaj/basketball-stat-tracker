"""
From a given CSV (generated from track_yoloe.py), export two CSVs containing shot data for later visualization.
"""

import statistics
from pathlib import Path

import cv2
import pandas as pd

TRACKS = "outputs/layups/tracks.csv"
VIDEO = "data/layups.mov"
FPS = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS)  


import statistics
from pathlib import Path
 
import cv2
import pandas as pd
 
TRACKS = "outputs/layups/tracks.csv"
VIDEO = "data/layups.mov"
FPS = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS)
 
# hoop
HOOP_SHAPE = (0.5, 0.8)     # height/width range for the hoop box; anything taller is probably the backboard
HOOP_SMOOTH_S = 0.25        # median window on each side of a frame, s
 
# ball tracking
HOOP_ZONE_U = (-1.5, 1.5)   # sideways range around the hoop center, hoop widths
HOOP_ZONE_V = (-1.5, 2.5)   # vertical range around the rim, hoop widths (negative is above)
FIXED_SPOT_SIZE = 0.1       # grid size for finding things that never move, hoop widths
FIXED_SHARE = 0.4           # a spot with a "ball" in more than this share of seconds is a fixed object
BALL_MAX_SPEED = 15         # hoop widths/s; shots hit the rim around 10
MAX_SPEED_CHANGE = 8        # how far off the predicted speed a match can be, hoop widths/s
BOX_JITTER = 0.3            # slack for boxes wobbling between frames, hoop widths
PATH_LOST_S = 0.5           # a path ends after going unseen this long, s
 
# shot attempts
RIM_V = 0.45                # how far below the rim still counts as rim height, hoop widths
RIM_U = 1.0                 # how far sideways from the hoop center, hoop widths
RIM_CONF = 0.2              # min detection confidence for an attempt
SAME_SHOT_S = 0.75          # rim-height moments closer than this are one shot (rim bounces), s
NET_EXIT_S = 1.5            # how long after the rim the ball has to come out of the net, s
 
# release
HOLD_MARGIN = 0.45          # how much to widen a person's box for the holding check, fraction of box width
RELEASE_LOOKBACK_S = 3      # how far back from the rim to look for the release, s
HOLD_MIN_S = 0.1            # ball has to be in someone's hands this long to count, s
 
# make and miss
SLOW_WINDOW_S = 0.2         # window after the net exit for measuring speed, s
SLOW_MIN_SEEN = 3           # need at least this many sightings in that window
SLOW_DROP =  4.5            # dropping slower than this means the net caught it, hoop widths/s
RIM_EDGE_U = 0.5            # exits farther sideways than this are outside the rim, hoop widths
NEAR_CENTER_U = 0.3         # exits closer than this are clearly through the middle, hoop widths
MIN_HIDDEN_FRAMES = 3       # frames the net has to hide the ball for
NET_BOTTOM_V = 0.55         # bottom of the net below the rim, hoop widths
 
# possession: skip balls at the feet, vote out flicker
POSSESSION_SMOOTH_S = 0.2  # majority vote window for possession, s on each side of a frame
HOLD_LOWEST = 0.9           # lowest point the ball can be in a person's box and still count as held, fraction of box height

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
        # difference normalized by width
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
        # subtracting margin from left, add it on right, same for top (not bottom bc dont want low possession)
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
        # speed is just from the last two points. divide by the actual time between them
        # since the path can skip frames when the ball isnt detected
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
            # no speed yet, so anything within max ball speed is fine
            return self.last.u, self.last.v, BOX_JITTER + BALL_MAX_SPEED * seconds
        # keep the last speed with some wiggle room, otherwise a still false detection can latch onto a ball flying past it
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
    # map the prompts that represent same idea into same idea
    prompt_to_kind = {"basketball hoop": "hoop", "basketball": "ball", "orange ball": "ball", "person": "person"}
    # nested dicts instead of a list per frame, since most frames dont have every kind
    detections = {"hoop": {}, "ball": {}, "person": {}}    # kind -> frame -> list of Box
    for row in df.to_dict("records"):
        kind = prompt_to_kind.get(row["class"])
        if kind is None:
            continue
        # setdefault makes an empty list the first time we see this frame, then appends to it after that
        # a frame can have more than one box per kind (two balls, a few people)
        box = Box(row["x1"], row["y1"], row["x2"], row["y2"], row["confidence"])
        detections[kind].setdefault(row["frame"], []).append(box)

    # frames start at 0, so the count is the last frame + 1
    # using the max instead of counting unique frames bc frames with no detections dont have a row at all
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
        # if every box that frame was the wrong shape, the frame just gets skipped
        if rim_shaped:
            # sort them using lambda bc confidence is feature
            best[frame] = max(rim_shaped, key=lambda box: box.confidence)

    # median over +/- HOOP_SMOOTH_S to cancel wobble, better than mean for smth like this
    window = to_frames(HOOP_SMOOTH_S)
    hoop = [None] * num_frames
    for frame in range(num_frames):
        nearby = [best[f] for f in range(frame - window, frame + window + 1) if f in best]
        if nearby:
            hoop[frame] = Box(*(statistics.median(getattr(box, edge) for box in nearby) for edge in ("x1", "y1", "x2", "y2")))
        elif frame > 0:
            # keep old hoop if no hoop valid
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
    # map spots to define set of seconds with ball in spot
    seconds_at_spot = {}
    for frame_balls in all_balls:
        for ball in frame_balls:
             # its a set so the same second only gets counted once
            seconds_at_spot.setdefault(ball.spot, set()).add(int(ball.frame / FPS))

    video_seconds = int(num_frames / FPS)
    fixed = set()
    for spot_u, spot_v in seconds_at_spot:
        # also check the 8 squares around it, otherwise something sitting right on a grid line
        seconds_near = set()
        for step_u in (-1, 0, 1):
            for step_v in (-1, 0, 1):
                # |= adds the neighbor's seconds into seconds_near (set union), and empty set if the neighbor never had a ball
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
    # need a hoop to get hoop width coords, so skip frames before it shows up
    # one list per frame. written this way bc [[]] * num_frames would make every frame share the same list
    # this was suggested by an LLM
    all_balls = [[] for _ in range(num_frames)]
    for frame, boxes in detections["ball"].items():
        if hoop[frame] is not None:
            all_balls[frame] = [Ball(frame, box, hoop[frame]) for box in boxes]
    # fixed spots need the whole video to find, so filter after building every frame
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
        # old paths stay in the list but only recent ones can take new balls
        active = [path for path in paths if path.seen_recently(frame)]

        # collect (candidate, path) pairs within the path's reach
        pairs = []
        for i, ball in enumerate(candidates[frame]):
            for j, path in enumerate(active):
                # gap goes first in the tuple so sorting the list sorts by distance
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

        # leftovers are either a new ball or a bad detection, give new path
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
    # leave frames with no ball as None
    hoop_ball = [None] * num_frames
    for path in paths:
        # loop by path instead of by frame since thats how the balls are stored
        for frame, ball in path.balls.items():
            # a frame can show up in more than one path, so keep whichever ball is more confident
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
        # compare to the end of the last shot, not the start
        # "shots and" to skip when there arent any shots yet
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
    held_until = None      # latest frame of the current holding streak, which is the release if long enough

    # path.balls only has frames where the ball was seen, so these arent always consecutive
    # reverse=True since we want earlier balls
    for frame in sorted(path.balls, reverse=True):
        # the path keeps going after the rim, skip those
        if frame > shot.rim_frame:
            continue
        if shot.rim_frame - frame > to_frames(RELEASE_LOOKBACK_S):
            return None
        # any() is true if at least one person in the frame is holding it
        if any(person.holds(path.balls[frame]) for person in detections["person"].get(frame, [])):
            # first held frame we hit going backwards = moment right before it left their hands
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

    previous = start_ball

    # find the frame where the ball comes out the bottom of the net
    # use hoop_ball, not the path bc net hiding ball breaks path
    for f in range(shot.rim_frame + 1, len(hoop_ball)):
        if f - shot.rim_end_frame > to_frames(NET_EXIT_S):
            break
        ball = hoop_ball[f]
        if ball is None:
            continue

        
        crossed_net = previous.v <= NET_BOTTOM_V and ball.v > NET_BOTTOM_V
        moving_down = ball.v > previous.v

        if crossed_net and moving_down:
            return f, ball.u

        # so that eventually the previous should cross the net
        previous = ball

    return None

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

    # look between the last frame at rim height and exit_frame
    hidden_count = 0
    max_hidden = 0

    at_rim = [f for f in range(shot.rim_frame, exit_frame) if hoop_ball[f] is not None and hoop_ball[f].v <= RIM_V]
    start = at_rim[-1] if at_rim else shot.rim_frame
    for f in range(start + 1, exit_frame):
        if hoop_ball[f] is None:
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
    # min so the window doesnt run past the end of the video
    window_end = min(len(hoop_ball), exit_frame + to_frames(SLOW_WINDOW_S))
    seen = []
    for f in range(exit_frame, window_end):
        if hoop_ball[f] is not None:
            seen.append(f)

    if len(seen) < SLOW_MIN_SEEN:
        return None

    # avg downward speed from first to last sighting in hoop wdiths
    first = hoop_ball[seen[0]]
    last = hoop_ball[seen[-1]]
    seconds = (last.frame - first.frame) / FPS
    speed = (last.v - first.v) / seconds
    # v goes down, so positive speed = falling
    return 0 <= speed < SLOW_DROP

def is_make(shot, hoop_ball):
    """Decides make or miss from the net clues.

    Args:
        shot: Shot to check.
        hoop_ball: From hoop_balls.

    Returns:
        True for a make, False for a miss.
    """
    # if doesnt go below net, then miss
    net_exit = leaves_net(shot, hoop_ball)
    if net_exit is None:
        return False
    exit_frame, exit_u = net_exit

    # if rlly far off, then miss
    if abs(exit_u) >= RIM_EDGE_U:
        return False

    # collect three checks (centered, hidden, slowed)
    near_center = abs(exit_u) < NEAR_CENTER_U
    hidden = net_hid_it(shot, hoop_ball, exit_frame, MIN_HIDDEN_FRAMES)
    slowed = net_slowed_it(hoop_ball, exit_frame)

    # need one of two position checks and speed check
    if slowed is None:
        # if no speed, make both position checks agree instead
        return near_center and hidden
    return slowed and (near_center or hidden)

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


def player_to_draw(detections, drawn_ball, num_frames):
    """Picks one player box per frame for the overlay.

    Takes the person nearest the drawn ball, so the highlight follows whoever is playing it;
    else the most confident person.

    Args:
        detections: From load_detections.
        drawn_ball: From ball_to_draw.
        num_frames: Frame count.

    Returns:
        List per frame of the person Box to draw, or None.
    """
    player = [None] * num_frames
    for frame in range(num_frames):
        people = detections["person"].get(frame, [])
        if not people:
            continue
        ball = drawn_ball[frame]
        if ball is not None:
            player[frame] = min(people, key=lambda person: abs(person.center_x - ball.x) + abs(person.center_y - ball.y))
        else:
            player[frame] = max(people, key=lambda person: person.confidence)
    return player


def export(out_dir, shots, hoop, drawn_ball, player, has_ball, num_frames):
    """Writes shots.csv (one row per shot) and frames.csv (one row per frame).

    Leaves cells blank where a value is unknown.

    Args:
        out_dir: Folder to write into, as a Path.
        shots: From find_shots, with results filled in.
        hoop: From steady_hoop.
        drawn_ball: From ball_to_draw.
        player: From player_to_draw.
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
        ball, h, person = drawn_ball[frame], hoop[frame], player[frame]
        frame_rows.append({
            "frame": frame,
            "ball_x": round(ball.x) if ball else None,
            "ball_y": round(ball.y) if ball else None,
            "hoop_x1": round(h.x1) if h else None,
            "hoop_y1": round(h.y1) if h else None,
            "hoop_x2": round(h.x2) if h else None,
            "hoop_y2": round(h.y2) if h else None,
            "player_x1": round(person.x1) if person else None,
            "player_y1": round(person.y1) if person else None,
            "player_x2": round(person.x2) if person else None,
            "player_y2": round(person.y2) if person else None,
            "possession": has_ball[frame],
            "fps": FPS,
        })
    whole = ["ball_x", "ball_y", "hoop_x1", "hoop_y1", "hoop_x2", "hoop_y2", "player_x1", "player_y1", "player_x2", "player_y2"]
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
    player = player_to_draw(detections, drawn_ball, num_frames)
    export(Path(TRACKS).parent, shots, hoop, drawn_ball, player, has_ball, num_frames)


if __name__ == "__main__":
    main()
