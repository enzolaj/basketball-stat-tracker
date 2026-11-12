## How to Run
1. Download or clone our repository into your workspace

2. Create virtual environment, activate it, and install required packages

        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

3. Find the path to your video and run the first file (track phase)

        python3 track_yoloe.py /path_to_video

    This might take a while.

4. Edit `TRACK` and `VIDEO` variables in `stat_tracker_alt.py` to match the paths for your video and track.csv. track.csv should generically be in `outputs/yoloe_bytetracks/tracks.csv`.

5. Run `stat_tracker_alt.py`

        python3 stat_tracker_alt.py

6. Run `overlay_from_csv.py` to get the final video. `/path_to_csv_folder` should generically be `outputs/yoloe_bytetrack/`

        python3 overlay_from_csv.py /path_to_video /path_to_csv_folder

7. Complete! Watch your analyzed video. in `outputs/yoloe_bytetrack/overlay_csv.mpy`