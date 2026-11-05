"""takes the csv (with bounding boxes of objects) and tracks the stats"""
import pandas as pd

def main():
    # check for ball and person interference
    # check for ball in the air
    # check for ball and basket intersection

    # read csv and save to dataframe
    initial_df = pd.read_csv('tracks.csv')
    print(initial_df.head(5))

    # find centers
    initial_df["center_x"] = (initial_df["x1"] + initial_df["x2"]) / 2
    initial_df["center_y"] = (initial_df["y1"] + initial_df["y2"]) / 2

    # extract ball and person and basket locations
    person = initial_df[initial_df["class"] == "person"].copy()
    ball = initial_df[initial_df["class"].isin(["orange ball", "basketball"])].copy()
    hoop = initial_df[initial_df["class"] == "basketball hoop"].copy()

    # create new ball positions df where there is nan when a ball is not detected
    ball_positions = ball[["frame", "center_x", "center_y", "confidence"]].copy()
    ball_positions = ball_positions.sort_values("frame")

    all_frames = pd.DataFrame({"frame": range(initial_df["frame"].min(), initial_df["frame"].max()+1)})
    ball_positions = all_frames.merge(ball_positions, on="frame", how="left")

    # 697 / 4967 unaccounted frames for basketball
    print(ball_positions.isna().sum())
    print(ball_positions["frame"].max())

    # if less than five frames, use linear interpolation, otherwise stays nan
    ball_positions[["center_x", "center_y"]] = (ball_positions[["center_x", "center_y"]].interpolate(limit=5, limit_area='inside'))

    # find velocity of ball
    fps = 60

    ball_positions["velocity_x"] = ball_positions["center_x"].diff() * fps
    ball_positions["velocity_y"] = ball_positions["center_y"].diff() * fps

    
    


    
    
    




if __name__ == "__main__":
    main()