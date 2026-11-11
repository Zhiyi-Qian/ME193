"""
Finger-count data collector (hands-free version).

Pick which numbers to collect on the command line. For each number the script
counts down, then automatically saves --count cropped images into data/<number>/,
then moves on to the next number. No key presses are needed.

Examples:
  python3 collect_data.py --camera 1 --person Zhiyi                 # all numbers 0-9
  python3 collect_data.py --camera 1 --person Zhiyi --labels 3      # just number 3
  python3 collect_data.py --camera 1 --person Zhiyi --labels 6 7 8  # several numbers
  python3 collect_data.py --camera 1 --person Zhiyi --labels 10     # "no hand" class
  python3 collect_data.py --camera 1 --person Zhiyi --reselect      # redraw detection area

Optional keys (click the video window first): space = pause/resume, q = quit.
"""
import argparse
import json
import os
import time

import cv2

ROI_FILE = "roi.json"
NAMES = {"10": "no hand (empty frame)"}


def nice(label):
    return NAMES.get(label, f"{label} fingers")


def parse_args():
    p = argparse.ArgumentParser(description="Collect finger-count training images")
    p.add_argument("--camera", type=int, default=0, help="camera index (try 0, 1, 2)")
    p.add_argument("--person", default="me", help="whose hand (goes in the file name)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="numbers to collect, e.g. --labels 3 or --labels 0 1 2 (default: all)")
    p.add_argument("--classes", type=int, default=11, help="total number of classes (0-9 fingers + 10 = no hand)")
    p.add_argument("--count", type=int, default=200, help="photos per number")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between photos")
    p.add_argument("--countdown", type=float, default=5, help="seconds to get ready before each number")
    p.add_argument("--out", default="data", help="output folder")
    p.add_argument("--reselect", action="store_true", help="redraw the detection area")
    return p.parse_args()


def load_roi():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE) as f:
            return tuple(json.load(f)["roi"])
    return None


def select_roi(frame):
    print("Click the window, drag a box just inside your tape frame, then press ENTER.")
    roi = cv2.selectROI("Select detection area", frame, showCrosshair=False)
    cv2.destroyWindow("Select detection area")
    if roi[2] == 0 or roi[3] == 0:
        return None
    roi = tuple(int(v) for v in roi)
    with open(ROI_FILE, "w") as f:
        json.dump({"roi": list(roi)}, f)
    print(f"Saved detection area {roi} to {ROI_FILE}")
    return roi


def put(img, text, pos, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    args = parse_args()
    labels = args.labels if args.labels else [str(i) for i in range(args.classes)]
    for lab in labels:
        if not lab.isdigit() or int(lab) >= args.classes:
            print(f"Label {lab} is not between 0 and {args.classes - 1}.")
            return
        os.makedirs(os.path.join(args.out, lab), exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Try another --camera number.")
        return
    # Fixed resolution so training and the UNO Q see the same image size.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ok, frame = cap.read()
    for _ in range(10):  # let the camera warm up / adjust exposure
        ok, frame = cap.read()
    if not ok:
        print("Camera opened but returned no image.")
        return
    print(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")

    roi = None if args.reselect else load_roi()
    while roi is None:
        roi = select_roi(frame)

    li = 0
    phase = "countdown"
    phase_start = time.time()
    taken = 0
    last_shot = 0.0
    paused = False
    quit_early = False

    print(f"Collecting {args.count} photos each for: {' '.join(labels)}")

    while li < len(labels):
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, w, h = roi
        crop = frame[y:y + h, x:x + w].copy()
        label = labels[li]
        now = time.time()

        if phase == "countdown":
            remaining = args.countdown - (now - phase_start)
            if remaining <= 0:
                phase, taken, last_shot = "capture", 0, 0.0
        elif not paused and now - last_shot >= args.interval:
            name = f"{label}_{args.person}_{int(now * 1000)}_{taken:03d}.png"
            cv2.imwrite(os.path.join(args.out, label, name), crop)
            taken += 1
            last_shot = now
            if taken >= args.count:
                total = len(os.listdir(os.path.join(args.out, label)))
                print(f"Done with {label}: {taken} new photos ({total} total in folder)")
                li += 1
                phase, phase_start = "countdown", now
                if li >= len(labels):
                    break
                continue

        display = frame.copy()
        if phase == "countdown":
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 255), 2)
            put(display, f"Get ready: show {nice(label)}", (10, 30), 0.9, (0, 200, 255))
            put(display, str(int(remaining) + 1), (display.shape[1] // 2 - 25, display.shape[0] // 2 + 25),
                2.5, (0, 200, 255), 5)
        else:
            color = (0, 200, 0) if paused else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            state = "PAUSED" if paused else "RECORDING"
            put(display, f"{state} {nice(label)}: {taken}/{args.count}", (10, 30), 0.9, color)
            bar_w = int((display.shape[1] - 20) * taken / args.count)
            cv2.rectangle(display, (10, 45), (10 + bar_w, 55), color, -1)
            tip = ("Keep the frame empty: try shadows, light changes, sleeve edges"
                   if label == "10" else "Move your hand slowly: tilt, shift, change fingers")
            put(display, tip, (10, display.shape[0] - 15), 0.5)
        put(display, f"Number {li + 1} of {len(labels)}", (display.shape[1] - 190, 30), 0.6)

        cv2.imshow("Collector", display)
        cv2.imshow("Crop (what the model sees)", crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit_early = True
            break
        if key == ord(" ") and phase == "capture":
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped early." if quit_early else "All done!")
    for lab in labels:
        print(f"  {lab}: {len(os.listdir(os.path.join(args.out, lab)))} photos")


if __name__ == "__main__":
    main()
