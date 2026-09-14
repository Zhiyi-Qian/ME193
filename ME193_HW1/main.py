"""
main.py

Gesture-controlled LEGO car.

    both arms up               -> forward
    left arm up                -> turn left
    right arm up                -> turn right
    both arms down              -> stop
    both arms out to the sides  -> backward
    right arm ~45 deg up        -> speed up
    left arm ~45 deg up         -> slow down

Usage:
    python main.py                 # connect to the real Double Motor hub
    python main.py --simulate      # no hardware needed, just prints commands
    python main.py --camera 1      # use a different webcam index
    Press 'q' or Esc in the video window to quit.
"""

import argparse
import sys

import cv2

from gesture_control import PoseGestureDetector
from car_controller import CarController

COMMAND_COLORS = {
    "FORWARD": (0, 200, 0),
    "BACKWARD": (0, 140, 255),
    "TURN_LEFT": (255, 200, 0),
    "TURN_RIGHT": (255, 200, 0),
    "STOP": (0, 0, 255),
    "SPEED_UP": (0, 255, 255),
    "SLOW_DOWN": (255, 0, 255),
    "HOLD": (180, 180, 180),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture-controlled LEGO car")
    parser.add_argument("--simulate", action="store_true",
                         help="Run without real hardware; just print the commands that would be sent.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    return parser.parse_args()


def draw_hud(frame, command, debug, speed):
    color = COMMAND_COLORS.get(command, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (30, 30, 30), -1)
    cv2.putText(frame, command, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"speed: {speed}", (frame.shape[1] - 220, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if debug.get("detected"):
        la, ra = debug["left_angle"], debug["right_angle"]
        info = f"L arm: {debug['left_state']} ({la:5.1f} deg)   R arm: {debug['right_state']} ({ra:5.1f} deg)"
        cv2.putText(frame, info, (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No person detected", (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.")
        sys.exit(1)

    detector = PoseGestureDetector()
    car = CarController(simulate=args.simulate)

    try:
        car.connect()
    except RuntimeError as exc:
        print(f"[car] {exc}")
        print("[car] Continuing in SIMULATE mode so you can still test the gestures.")
        car = CarController(simulate=True)

    print("Press 'q' or Esc in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            # Run pose estimation on the RAW (un-flipped) frame so MediaPipe's
            # left/right landmarks match the person's actual left/right arm.
            annotated, command, debug = detector.process(frame)
            car.apply(command)

            draw_hud(annotated, command, debug, car.speed)
            # Flip only for the on-screen preview, so it feels like a mirror.
            preview = cv2.flip(annotated, 1)
            cv2.imshow("Gesture Car Control", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break
    finally:
        car.apply("STOP")
        car.disconnect()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
