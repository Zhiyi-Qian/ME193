"""
Finger lock - computer version with a web interface.

The laptop runs the camera, the neural network and the lock policy, and serves a
web page (lock_ui.html) with the live camera view and step-by-step instructions.
The UNO Q app handles the physical button, LED and DC motor latch.

Run:
  python3 lock_web.py --camera 0 --code 3 1 5 --board http://10.247.137.113:7000
Then open http://localhost:8000 in your browser. Stop with Ctrl+C.

Options:
  --window        also show the OpenCV window (keys: b = button, q = quit)
  --show-digits   show the entered digits on the page instead of dots
  --port 8000     web page port
  --no-viz        turn off the heatmap / feature maps (saves CPU, e.g. on the UNO Q)
"""
import argparse
import base64
import json
import os
import threading
import time
import urllib.request
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from train import FingerNet

HERE = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(HERE, "lock_ui.html")

shared_lock = threading.Lock()
shared = {"state": {}, "press": False}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--code", type=int, nargs="+", required=True, help="e.g. --code 3 1 5")
    p.add_argument("--board", default=None, help="UNO Q address, e.g. http://10.247.137.113:7000")
    p.add_argument("--model", default="finger_model.pth")
    p.add_argument("--stable-frames", type=int, default=8)
    p.add_argument("--min-conf", type=float, default=0.8)
    p.add_argument("--timeout", type=float, default=15)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--window", action="store_true")
    p.add_argument("--show-digits", action="store_true")
    p.add_argument("--no-viz", action="store_true", help="skip heatmap and feature maps")
    p.add_argument("--viz-every", type=int, default=3, help="update visuals every N frames")
    return p.parse_args()


# ------------------------------------------------------------------ "what the network sees"
def to_b64_jpg(img, quality=80):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(jpg.tobytes()).decode() if ok else None


def forward_with_maps(model, x):
    """Run the CNN one layer at a time and keep the output of each ReLU (= each conv layer)."""
    acts = []
    h = x
    for layer in model.features:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            acts.append(h)
    return model.classifier(h), acts


def feature_grid(act, n=9, tile=64):
    """Show the n most active channels of one layer as a 3 x 3 grid of small images."""
    a = act[0].detach()
    idx = a.mean(dim=(1, 2)).argsort(descending=True)[:n]
    tiles = []
    for i in idx:
        m = a[i].numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        m = cv2.resize(np.uint8(m * 255), (tile, tile), interpolation=cv2.INTER_NEAREST)
        tiles.append(cv2.applyColorMap(m, cv2.COLORMAP_VIRIDIS))
    while len(tiles) < n:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    gap = 4
    cols = 3
    rows = (n + cols - 1) // cols
    grid = np.full((rows * tile + (rows - 1) * gap, cols * tile + (cols - 1) * gap, 3), (122, 38, 24), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[r * (tile + gap):r * (tile + gap) + tile, c * (tile + gap):c * (tile + gap) + tile] = t
    return grid


def visualize(model, x, crop, pred):
    """Grad-CAM heatmap on the last conv layer, plus feature maps of all 3 conv layers."""
    logits, acts = forward_with_maps(model, x)
    last = acts[-1]
    grads = torch.autograd.grad(logits[0, pred], last)[0][0]      # d(score) / d(activation)
    weights = grads.mean(dim=(1, 2))                                  # importance of each channel
    cam = torch.relu((weights[:, None, None] * last[0]).sum(0)).detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    ch, cw = crop.shape[:2]
    scale = 240 / max(ch, cw)
    view = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (view.shape[1], view.shape[0]))),
                             cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(view, 0.55, heat, 0.45, 0)

    size = x.shape[-1]
    net_input = cv2.resize(crop, (size, size))                        # what the network actually gets
    layers = []
    for a in acts:
        layers.append({"img": to_b64_jpg(feature_grid(a)),
                       "channels": int(a.shape[1]), "size": int(a.shape[2])})
    return {
        "input": to_b64_jpg(cv2.resize(net_input, (192, 192), interpolation=cv2.INTER_NEAREST)),
        "input_size": int(size),
        "layers": layers,
        "heatmap": to_b64_jpg(overlay),
    }


# ------------------------------------------------------------------ UNO Q link
class Board:
    def __init__(self, url):
        self.url = url.rstrip("/") if url else None
        self.online = None

    def _get(self, path):
        if not self.url:
            return {}
        try:
            with urllib.request.urlopen(self.url + path, timeout=1.0) as r:
                self.online = True
                body = r.read().decode()
                return json.loads(body) if body else {}
        except Exception:
            self.online = False
            return {}

    def status(self):
        if not self.url:
            return "none"
        return {True: "online", False: "offline"}.get(self.online, "unknown")

    def button(self):
        return bool(self._get("/api/button").get("pressed")) if self.url else False

    def leds(self, on):
        self._get("/api/leds_on" if on else "/api/leds_off")

    def unlock(self):
        self._get("/api/unlock")

    def lock(self):
        self._get("/api/lock")

    def fail(self):
        self._get("/api/fail")


# ------------------------------------------------------------------ web server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(UI_FILE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, b"lock_ui.html is missing - put it next to lock_web.py", "text/plain")
        elif self.path.startswith("/api/state"):
            with shared_lock:
                body = json.dumps(shared["state"]).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/api/press"):
            with shared_lock:
                shared["press"] = True
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def start_server(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------ main loop
def main():
    args = parse_args()
    code = args.code

    ckpt = torch.load(args.model, map_location="cpu")
    n_classes = ckpt["classes"]
    names = ckpt.get("names", [str(c) for c in range(n_classes)])
    blank = 10 if n_classes > 10 else None          # the "no hand" class, if the model has one
    clear_set = {0} if blank is None else {0, blank}
    model = FingerNet(n_classes, ckpt["size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((ckpt["size"], ckpt["size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    with open(os.path.join(HERE, "roi.json")) as f:
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

    start_server(args.port)
    print(f"Lock page: http://localhost:{args.port}   (Ctrl+C to stop)")

    recent = deque(maxlen=args.stable_frames)
    state = "IDLE"
    entered = []
    need_clear = True
    last_activity = 0.0
    last_poll = 0.0
    fails = 0
    frame_no = 0
    viz = None
    event = {"id": 0, "kind": "ready", "time": time.time()}

    def new_event(kind):
        event["id"] += 1
        event["kind"] = kind
        event["time"] = time.time()
        print("Event:", kind)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            fh, fw = frame.shape[:2]
            crop = frame[y:y + h, x:x + w].copy()
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            xin = tf(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(xin), dim=1)[0]
            conf, pred = probs.max(0)
            conf, pred = conf.item(), pred.item()
            frame_no += 1
            if not args.no_viz and frame_no % args.viz_every == 0:
                try:
                    viz = visualize(model, xin, crop, pred)
                except Exception as e:
                    print("Visualization error:", e)

            recent.append(pred if conf >= args.min_conf else None)
            stable = None
            if len(recent) == recent.maxlen:
                value, n = Counter(recent).most_common(1)[0]
                if value is not None and n == recent.maxlen:
                    stable = value
            hold = sum(1 for r in recent if r is not None and r == pred) / recent.maxlen

            now = time.time()
            pressed = False
            with shared_lock:
                if shared["press"]:
                    pressed = shared["press"] and state in ("IDLE", "OPEN")
                    shared["press"] = False
            if args.window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b") and state in ("IDLE", "OPEN"):
                    pressed = True
            if not pressed and state in ("IDLE", "OPEN") and now - last_poll >= 0.2:
                last_poll = now
                pressed = board.button()

            # -------------------------------------------------- lock policy
            if state == "IDLE":
                if pressed:
                    state, entered, need_clear = "READING", [], True
                    recent.clear()
                    last_activity = now
                    board.leds(True)
                    new_event("started")

            elif state == "READING":
                if now - last_activity > args.timeout:
                    state = "IDLE"
                    board.leds(False)
                    new_event("timeout")
                elif stable is not None:
                    if need_clear:
                        if stable in clear_set:
                            need_clear = False
                            recent.clear()
                            last_activity = now
                    elif stable not in clear_set:
                        entered.append(stable)
                        need_clear = True
                        recent.clear()
                        last_activity = now
                        if len(entered) == len(code):
                            if entered == code:
                                state = "OPEN"
                                fails = 0
                                board.unlock()
                                new_event("unlocked")
                            else:
                                state = "IDLE"
                                fails += 1
                                board.fail()
                                new_event("wrong")
                        else:
                            new_event("accepted")

            elif state == "OPEN":
                if pressed:
                    state = "IDLE"
                    board.lock()
                    board.leds(False)
                    new_event("locked")

            # -------------------------------------------------- share with page
            small = cv2.resize(frame, (480, int(480 * fh / fw)))
            ok_jpg, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            timeout_left = None
            if state == "READING":
                timeout_left = max(0.0, args.timeout - (now - last_activity))
            snapshot = {
                "state": state,
                "code_len": len(code),
                "entered": len(entered),
                "digits": entered if args.show_digits else None,
                "need_clear": need_clear,
                "pred": pred,
                "conf": round(conf, 3),
                "probs": [round(p, 3) for p in probs.tolist()],
                "hold": round(hold, 3),
                "timeout_left": timeout_left,
                "timeout_total": args.timeout,
                "board": board.status(),
                "fails": fails,
                "event": {"id": event["id"], "kind": event["kind"], "age": round(now - event["time"], 2)},
                "roi": [x / fw, y / fh, w / fw, h / fh],
                "names": names,
                "blank": blank,
                "viz": viz,
                "frame": base64.b64encode(jpg.tobytes()).decode() if ok_jpg else None,
            }
            with shared_lock:
                shared["state"] = snapshot

            if args.window:
                display = frame.copy()
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 111, 181), 2)
                cv2.putText(display, f"{state}  guess {pred} ({conf:.0%})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Finger lock", display)
    except KeyboardInterrupt:
        pass
    finally:
        board.lock()
        board.leds(False)
        cap.release()
        cv2.destroyAllWindows()
        print("Lock stopped.")


if __name__ == "__main__":
    main()
