"""
Finger lock - computer version.

The laptop runs the camera and the neural network and makes the lock decisions.
The UNO Q (running unoq_app in App Lab) handles the button, LED and DC motor latch.
(Simple OpenCV-window version. lock_web.py is the full version with the web interface.)

Lock policy:
  IDLE     -> button pressed -> LEDs on, start READING
  READING  -> show 0 (fist / empty), then a digit; a digit counts only when the
              model gives the same answer with high confidence for several frames.
              Repeat for each digit. Wrong code -> red flash, back to IDLE.
              Right code -> motor opens the latch (OPEN). No new digit for --timeout s -> IDLE.
  OPEN     -> button pressed -> motor closes the latch, back to IDLE

Run with the board:     python3 lock.py --camera 0 --code 3 1 5 --board http://192.168.1.23:7000
Run without the board:  python3 lock.py --camera 0 --code 3 1 5     (press b for the button)
Keys (click the video window first): b = button, q = quit
"""
import argparse
import json
import time
import urllib.request
from collections import Counter, deque

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train import FingerNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--code", type=int, nargs="+", required=True, help="e.g. --code 3 1 5")
    p.add_argument("--board", default=None, help="UNO Q address, e.g. http://192.168.1.23:7000")
    p.add_argument("--model", default="finger_model.pth")
    p.add_argument("--stable-frames", type=int, default=8)
    p.add_argument("--min-conf", type=float, default=0.8)
    p.add_argument("--timeout", type=float, default=15, help="seconds without a new digit before reset")
    p.add_argument("--show-digits", action="store_true", help="show entered digits instead of *")
    return p.parse_args()


class Board:
    """Talks to the UNO Q app over HTTP. Without an address it just prints."""

    def __init__(self, url):
        self.url = url.rstrip("/") if url else None
        self.online = None

    def _get(self, path):
        if not self.url:
            print(f"[no board] {path}")
            return {}
        try:
            with urllib.request.urlopen(self.url + path, timeout=1.0) as r:
                self.online = True
                body = r.read().decode()
                return json.loads(body) if body else {}
        except Exception:
            self.online = False
            return {}

    def button(self):
        if not self.url:
            return False
        return bool(self._get("/api/button").get("pressed"))

    def leds(self, on):
        self._get("/api/leds_on" if on else "/api/leds_off")

    def unlock(self):
        self._get("/api/unlock")

    def lock(self):
        self._get("/api/lock")

    def fail(self):
        self._get("/api/fail")


def main():
    args = parse_args()
    code = args.code

    ckpt = torch.load(args.model, map_location="cpu")
    model = FingerNet(ckpt["classes"], ckpt["size"])
    clear_set = {0, 10} if ckpt["classes"] > 10 else {0}   # fist or "no hand" separates digits
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((ckpt["size"], ckpt["size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    with open("roi.json") as f:
        x, y, w, h = json.load(f)["roi"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    board = Board(args.board)
    board.lock()
    board.leds(False)

    recent = deque(maxlen=args.stable_frames)
    state = "IDLE"
    entered = []
    need_clear = True
    last_activity = 0.0
    last_poll = 0.0
    message, message_until = "", 0.0

    def say(text, seconds=3.0):
        nonlocal message, message_until
        message, message_until = text, time.time() + seconds
        print(text)

    print(f"Lock ready. Code length {len(code)}. Press the button (or b) to start.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y:y + h, x:x + w].copy()
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
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        pressed = key == ord("b")
        if not pressed and state in ("IDLE", "OPEN") and now - last_poll >= 0.2:
            last_poll = now
            pressed = board.button()

        # ------------------------------------------------------------ lock policy
        if state == "IDLE":
            if pressed:
                state, entered, need_clear = "READING", [], True
                recent.clear()
                last_activity = now
                board.leds(True)
                say("Listening... show your code")

        elif state == "READING":
            if now - last_activity > args.timeout:
                state = "IDLE"
                board.leds(False)
                say("Timed out - press the button to try again")
            elif stable is not None:
                if need_clear:
                    if stable in clear_set:          # hand cleared: ready for the next digit
                        need_clear = False
                        recent.clear()
                elif stable not in clear_set:   # a new digit was shown
                    entered.append(stable)
                    need_clear = True
                    recent.clear()
                    last_activity = now
                    say(f"Digit {len(entered)} accepted", 1.5)
                    if len(entered) == len(code):
                        if entered == code:
                            state = "OPEN"
                            board.unlock()
                            say("UNLOCKED - press the button to lock again", 10)
                        else:
                            state = "IDLE"
                            board.fail()
                            say("Wrong code - still locked")

        elif state == "OPEN":
            if pressed:
                state = "IDLE"
                board.lock()
                board.leds(False)
                say("Locked")

        # ------------------------------------------------------------ display
        display = frame.copy()
        colors = {"IDLE": (160, 160, 160), "READING": (0, 200, 255), "OPEN": (0, 200, 0)}
        color = colors[state]
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        def put(text, pos, scale=0.7, c=(255, 255, 255), t=2):
            cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 3)
            cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, c, t)

        put(f"State: {state}", (10, 30), 0.8, color)
        slots = [str(d) if args.show_digits else "*" for d in entered]
        slots += ["_"] * (len(code) - len(entered))
        put("Code: " + " ".join(slots), (10, 65), 0.8, color)
        if state == "READING":
            hint = "Clear the frame (fist or no hand)" if need_clear else f"Show digit {len(entered) + 1}"
            put(hint, (10, 100), 0.7)
        put(f"Model: {pred} ({conf:.0%})", (10, display.shape[0] - 45), 0.6)
        if args.board:
            status = "board online" if board.online else "board OFFLINE" if board.online is False else "board ?"
            put(status, (display.shape[1] - 170, 30), 0.55)
        if now < message_until:
            put(message, (10, display.shape[0] - 15), 0.65, (0, 255, 255))

        cv2.imshow("Finger lock", display)

    board.lock()
    board.leds(False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
