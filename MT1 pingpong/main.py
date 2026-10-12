"""
LEGO Ping-Pong - run this to play.

    python main.py            # with the Double Motor
    python main.py --fake     # no motor: press SPACE to swing
    python main.py --camera 1 # use a different camera

Keys: C = serve, S = start, 1/2/3 = level, SPACE = fake swing, Q = quit
"""

import argparse
import json
import os
import tempfile
import time
from datetime import date

import cv2
import numpy as np

import config
from audio_trigger import ClapDetector
from game import INCOMING, OUTGOING, READY, WAIT_START, Game
from paddle import FakePaddle, Paddle
from score_mqtt import ScorePublisher
from swing import SwingDetector
from vision import Tracker, level_from_tag_size

WHITE, GREEN, RED, YELLOW, CYAN, ORANGE = (
    (255, 255, 255), (80, 220, 80), (60, 60, 230), (0, 220, 255), (230, 200, 60), (0, 140, 255))


def px(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)


def text(img, s, pos, scale=0.8, color=WHITE, thick=2):
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def load_daily_record(path=config.DAILY_RECORD_PATH, today=None):
    day = (today or date.today()).isoformat()
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as record_file:
        saved = json.load(record_file)
    if saved.get("date") != day:
        return 0
    record = saved["record"]
    if type(record) is not int or record < 0:
        raise ValueError(f"Invalid daily record in {path}: {record!r}")
    return record


def save_daily_record(record, path=config.DAILY_RECORD_PATH, today=None):
    if type(record) is not int or record < 0:
        raise ValueError(f"Daily record must be a non-negative integer: {record!r}")
    day = (today or date.today()).isoformat()
    directory = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         delete=False) as record_file:
            temp_path = record_file.name
            json.dump({"date": day, "record": record}, record_file)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def on_mouse(event, x, y, _flags, state):
    if event == cv2.EVENT_LBUTTONUP:
        state["point"] = (x, y)


def court_geometry(zone_points):
    left, right = zone_points
    top_y, net_y = 0.12, 0.40
    top_left_x = 0.5 + (left[0] - 0.5) * 0.35
    top_right_x = 0.5 + (right[0] - 0.5) * 0.35
    top_left = (top_left_x, top_y)
    top_right = (top_right_x, top_y)

    def at_net(top, bottom):
        ratio = (net_y - top_y) / max(bottom[1] - top_y, 1e-6)
        ratio = min(max(ratio, 0.0), 1.0)
        return top[0] + (bottom[0] - top[0]) * ratio, net_y

    return (top_left, top_right, right, left), (at_net(top_left, left),
                                                 at_net(top_right, right))


def draw(img, game, info, now, last_swing, mic):
    h, w = img.shape[:2]

    # The sidelines terminate at the tag centers, so their motion is visible.
    table, net = court_geometry(game.zone_points)
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array([px(p, w, h) for p in table])], (90, 60, 20))
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    edge_color = CYAN if game.zone_from_tags else ORANGE
    points = [px(p, w, h) for p in table]
    cv2.polylines(img, [np.array(points)], True, edge_color, 4, cv2.LINE_AA)
    cv2.line(img, px(net[0], w, h), px(net[1], w, h), WHITE, 2)
    if not game.zone_from_tags:
        text(img, "Zone tags (0 and 1) not seen - using default zone", (10, h - 15), 0.55, ORANGE, 1)
    else:
        for tag_id, point in game.zone_tag_points.items():
            center = px(point, w, h)
            cv2.circle(img, center, 12, GREEN, -1, cv2.LINE_AA)
            cv2.circle(img, center, 15, WHITE, 2, cv2.LINE_AA)
            text(img, f"TAG {tag_id}", (center[0] + 14, center[1] - 10),
                 0.5, WHITE, 1)

    # Opponent paddle
    ox = int(game.opp_x * w)
    cv2.rectangle(img, (ox - 35, int(0.13 * h)), (ox + 35, int(0.16 * h)), RED, -1)

    # Ball + where it will land (so you know where to put your hand)
    ball = game.ball_screen(now)
    if ball and game.state == INCOMING:
        cv2.circle(img, px(game.ball["to"], w, h), int(config.HIT_RADIUS * w), YELLOW, 1)
        cv2.circle(img, px(game.ball["to"], w, h), int(config.SWEET_RADIUS * w), YELLOW, 1)
    if ball:
        cv2.circle(img, px(ball[:2], w, h), int(ball[2] * w), (240, 240, 240), -1)
        cv2.circle(img, px(ball[:2], w, h), int(ball[2] * w), (0, 0, 0), 1)

    # Your hand (the "paddle")
    if info["wrist"]:
        col = GREEN if game.in_zone(info["wrist"]) else RED
        cv2.circle(img, px(info["wrist"], w, h), 18, col, 3)
        if info["elbow"]:
            cv2.line(img, px(info["elbow"], w, h), px(info["wrist"], w, h), col, 3)

    # HUD
    text(img, f"Streak {game.streak}   Record {game.record}   Level {game.level}", (10, 35), 0.9, CYAN)
    if game.message:
        text(img, game.message, (10, 70), 0.8, YELLOW)
    if game.last_result:
        good = game.last_result in ("Sweet!", "SMASH!", "Edge...")
        text(img, game.last_result, (10, 105), 0.9, GREEN if good else RED)
    if last_swing and now - last_swing[0] < 1.0:
        text(img, f"swing: {last_swing[1]}", (w - 230, 35), 0.8, ORANGE)
    if mic.ok:
        bar = int(min(mic.level / max(config.CLAP_MIN_RMS * 2, 1e-6), 1.0) * 120)
        cv2.rectangle(img, (w - 140, 50), (w - 140 + bar, 62), GREEN, -1)
        cv2.rectangle(img, (w - 140, 50), (w - 20, 62), WHITE, 1)


def draw_round_screen(img, phase, remaining, round_record, daily_record):
    h, w = img.shape[:2]
    if phase == "playing":
        text(img, f"ROUND  {max(0, int(remaining + 0.999))}s",
             (w - 200, 85), 0.9, CYAN)
        return None

    panel_w, panel_h = min(900, w - 40), min(450, h - 40)
    left, top = (w - panel_w) // 2, (h - panel_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (left + panel_w, top + panel_h),
                  (25, 25, 25), -1)
    img[:] = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)

    if phase == "instructions":
        title = "READY FOR PING-PONG?"
        lines = [
            "Each round lasts 30 seconds.",
            "Move your tracked wrist toward the yellow ball target.",
            "Swing the motor as the ball arrives.",
            "Inner circle = sweet spot; outer circle = hit limit.",
            "Press C or clap to serve. Hold AprilTag 3 to start.",
            "Click START or press S/ENTER to begin.",
        ]
        button_label = "START ROUND"
    else:
        title = "ROUND OVER"
        lines = [
            f"Best streak this round: {round_record}",
            f"Best streak today: {daily_record}",
        ]
        button_label = "PLAY AGAIN"

    text(img, title, (left + 28, top + 58), 1.1, CYAN)
    line_y = top + 112
    for line in lines:
        text(img, line, (left + 30, line_y), 0.65, WHITE, 1)
        line_y += 42
    button_w, button_h = min(300, panel_w - 60), 52
    button_x = left + (panel_w - button_w) // 2
    button_y = top + panel_h - 105
    cv2.rectangle(img, (button_x, button_y),
                  (button_x + button_w, button_y + button_h), (40, 125, 45), -1)
    cv2.rectangle(img, (button_x, button_y),
                  (button_x + button_w, button_y + button_h), GREEN, 2)
    label_size, _ = cv2.getTextSize(button_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_x = button_x + (button_w - label_size[0]) // 2
    label_y = button_y + (button_h + label_size[1]) // 2
    text(img, button_label, (label_x, label_y), 0.7, WHITE, 2)
    footer = ("Press Q to quit" if phase == "instructions"
              else "Click PLAY AGAIN or press R/ENTER  |  Q to quit")
    text(img, footer, (left + 28, top + panel_h - 18), 0.55, YELLOW, 1)
    return button_x, button_y, button_x + button_w, button_y + button_h


class NoMic:
    ok, level = False, 0.0

    def clapped(self):
        return False

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="play without the motor")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    daily_record = load_daily_record()
    paddle = FakePaddle() if args.fake else Paddle()
    paddle.connect()
    detector = SwingDetector()
    tracker = Tracker()
    mic = ClapDetector() if config.AUDIO_ENABLED else NoMic()
    mqtt_pub = ScorePublisher()
    mqtt_pub.publish(0.0)

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera not found. Try --camera 1")
    game = Game(aspect=frame.shape[0] / frame.shape[1])
    daily_record_day = date.today()
    phase = "instructions"
    round_started_at = None
    round_record = 0
    last_swing = None
    key_events = []          # events from key presses, handled next frame
    t_start = time.time()
    mouse_state = {"point": None}
    window_name = "LEGO Ping-Pong"

    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse, mouse_state)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            current_day = date.today()
            if current_day != daily_record_day:
                daily_record_day = current_day
                daily_record = load_daily_record(today=current_day)
            img, info = tracker.process(frame, (now - t_start) * 1000)

            # IMU -> swings
            for t, label in detector.update(paddle.get_samples(since=now - 2)):
                game.add_swing(t, label)
                last_swing = (now, label)

            # AprilTag level (only before/between rallies)
            events, key_events = key_events, []
            lvl_tag = info["tags"].get(config.TAG_LEVEL)
            if lvl_tag and game.state in (WAIT_START, READY):
                events += game.set_level(level_from_tag_size(lvl_tag["size"]))

            if phase == "instructions":
                events += game.update(now, info["wrist"], info["tags"])
                if any(ev == "start" for ev, _ in events):
                    phase = "playing"
                    round_started_at = now
                    round_record = 0
                    game.swings.clear()
            elif phase == "playing":
                if mic.clapped():
                    events += game.serve(now)
                events += game.update(now, info["wrist"], info["tags"])
                round_record = max(round_record, game.streak)
                if round_record > daily_record:
                    daily_record = round_record
                    save_daily_record(daily_record, today=daily_record_day)
                if now - round_started_at >= config.ROUND_DURATION_S:
                    phase = "results"

            # Events -> haptics, lights, beeps, MQTT
            for ev, val in events:
                if ev == "hit":
                    quality, label, streak = val
                    paddle.haptic("smash" if label == "smash" else quality)
                    paddle.light("green" if quality == "sweet" else "yellow")
                    paddle.beep(min(440 * 2 ** (streak / 12), 2000))   # pitch climbs with your combo
                elif ev == "miss":
                    paddle.haptic("miss")
                    paddle.light("red")
                    paddle.beep(180)
                elif ev == "record":
                    mqtt_pub.publish(val)
                elif ev == "warn":
                    paddle.haptic("warn")
                elif ev == "opp_miss":
                    paddle.light("purple", pattern=3)
                    paddle.beep(880, pattern=3)
                elif ev in ("start", "level"):
                    paddle.light({1: "green", 2: "yellow", 3: "red"}[game.level])
                    paddle.beep(523 + 100 * game.level, pattern=1)
                elif ev == "serve":
                    paddle.beep(660)
            mqtt_pub.tick()

            draw(img, game, info, now, last_swing, mic)
            if phase == "results":
                remaining = 0
            elif phase == "playing":
                remaining = config.ROUND_DURATION_S - (now - round_started_at)
            else:
                remaining = config.ROUND_DURATION_S
            restart_button = draw_round_screen(img, phase, remaining,
                                               round_record, daily_record)
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF
            click = mouse_state["point"]
            mouse_state["point"] = None
            if (restart_button and click
                    and restart_button[0] <= click[0] <= restart_button[2]
                    and restart_button[1] <= click[1] <= restart_button[3]):
                key = ord("s") if phase == "instructions" else ord("r")
            if key == ord("q"):
                break
            elif phase == "instructions" and key in (ord("s"), 13, 10):
                if game.state == WAIT_START:
                    key_events += game.start()
                phase = "playing"
                round_started_at = time.time()
                round_record = 0
                game.swings.clear()
            elif phase == "results" and key in (ord("r"), 13, 10):
                game = Game(brain=game.brain, aspect=game.aspect)
                phase = "instructions"
                round_started_at = None
                round_record = 0
                last_swing = None
            elif phase == "playing" and key == ord("c"):
                key_events += game.serve(now)
            elif phase == "playing" and key in (ord("1"), ord("2"), ord("3")):
                key_events += game.set_level(key - ord("0"))
            elif phase == "playing" and key == ord(" ") and args.fake:
                paddle.fake_swing()
    finally:
        game.brain.save()
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        mic.close()
        mqtt_pub.close()
        paddle.disconnect()


if __name__ == "__main__":
    main()
