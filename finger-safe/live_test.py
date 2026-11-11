"""
Live test of the finger-count model on the webcam.

Uses the same detection area (roi.json) and image processing as training, shows the
model's guess and confidence for every frame, plus a "stable" answer that only
appears when the last several frames agree (this is how the lock will decide).

Run:  python3 live_test.py --camera 0
Keys (click the video window first): q = quit, s = save a snapshot of the crop
"""
import argparse
import json
import os
import time
from collections import Counter, deque

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train import FingerNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="finger_model.pth")
    p.add_argument("--stable-frames", type=int, default=8, help="frames that must agree")
    p.add_argument("--min-conf", type=float, default=0.7, help="minimum confidence to count a frame")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    n_classes, size = ckpt["classes"], ckpt["size"]
    model = FingerNet(n_classes, size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open("roi.json") as f:
        x, y, w, h = json.load(f)["roi"]

    # Must match the validation transform in train.py exactly.
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recent = deque(maxlen=args.stable_frames)
    fps_t, fps = time.time(), 0.0
    os.makedirs("snapshots", exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y:y + h, x:x + w].copy()

        # OpenCV gives BGR; training images were loaded as RGB by PIL.
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]
        conf, pred = probs.max(0)
        conf, pred = conf.item(), pred.item()

        recent.append(pred if conf >= args.min_conf else None)
        stable = None
        if len(recent) == recent.maxlen:
            value, n = Counter(recent).most_common(1)[0]
            if value is not None and n == recent.maxlen:
                stable = value

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
        fps_t = now

        display = frame.copy()
        color = (0, 200, 0) if stable is not None else (0, 200, 255)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        def put(text, pos, scale=0.7, c=(255, 255, 255), t=2):
            cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 3)
            cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, c, t)

        put(f"Guess: {pred}  ({conf:.0%})", (10, 30), 0.9, color)
        put(f"Stable: {stable if stable is not None else '-'}", (10, 65), 0.9, color)
        put(f"{fps:.0f} fps", (display.shape[1] - 90, 30), 0.6)

        # probability bar for every class
        bx, by, bw = 10, display.shape[0] - 20, 28
        for c in range(n_classes):
            bh = int(probs[c].item() * 80)
            col = (0, 200, 0) if c == pred else (200, 200, 200)
            cv2.rectangle(display, (bx + c * (bw + 4), by - bh), (bx + c * (bw + 4) + bw, by), col, -1)
            put(str(c), (bx + c * (bw + 4) + 8, by + 15), 0.45, t=1)

        cv2.imshow("Live test", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join("snapshots", f"pred{pred}_{int(now * 1000)}.png")
            cv2.imwrite(path, crop)
            print("Saved", path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
