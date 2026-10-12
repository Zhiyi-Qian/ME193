"""
Record labeled swings to train the swing classifier.

    python collect_swings.py                 # 20 of each label
    python collect_swings.py --reps 30

For each recording: the paddle beeps, you have ~1.5 s to do the move.
For "none", wave the paddle around casually / walk / scratch your head -
anything that is NOT a real swing. That teaches the model what to ignore.

Each recording is saved as one line of JSON in swing_data.jsonl, so you can
stop any time and run it again later to add more.
"""

import argparse
import json
import random
import time

import config
from paddle import Paddle
from swing import LABELS, WINDOW_LEN, gyro_mag


def cut_window(samples):
    """Window of WINDOW_LEN samples around the biggest gyro peak."""
    if len(samples) < WINDOW_LEN:
        return None
    i = max(range(len(samples)), key=lambda k: gyro_mag(samples[k]))
    i = min(max(i, config.WINDOW_BEFORE), len(samples) - config.WINDOW_AFTER - 1)
    return samples[i - config.WINDOW_BEFORE: i + config.WINDOW_AFTER + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    paddle = Paddle()
    paddle.connect()

    plan = [lbl for lbl in LABELS for _ in range(args.reps)]
    random.shuffle(plan)   # mixed order = less "getting tired" bias per label
    print(f"\nRecording {len(plan)} moves. Ctrl+C to stop early (data is kept).\n")

    try:
        with open(config.DATA_PATH, "a") as f:
            for n, label in enumerate(plan, 1):
                input(f"[{n}/{len(plan)}] Next: {label.upper():9s} - press Enter, "
                      f"then move after the beep ")
                paddle.beep(660)
                t0 = time.time()
                time.sleep(1.5)
                window = cut_window(paddle.get_samples(since=t0))
                if window is None:
                    print("   not enough IMU data - is the motor connected?")
                    continue
                peak = max(gyro_mag(s) for s in window)
                print(f"   saved. peak |gyro| = {peak:.0f}")
                f.write(json.dumps({"label": label, "window": window}) + "\n")
                f.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        paddle.disconnect()
    print(f"Data saved to {config.DATA_PATH}. Now run: python train_swing_model.py")


if __name__ == "__main__":
    main()
