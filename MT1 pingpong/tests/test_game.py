"""
Unit tests - no camera, motor or microphone needed.
Run from the project folder:   python -m pytest -q
"""

import io
import os
import random
import ssl
import sys
import tempfile
from datetime import date
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from game import INCOMING, OUTGOING, READY, WAIT_START, Game, OpponentBrain  # noqa: E402
from paddle import FakePaddle  # noqa: E402
from main import (court_geometry, draw_round_screen, load_daily_record,
                  save_daily_record)  # noqa: E402
from swing import SwingDetector, SwingClassifier, rule_based_label  # noqa: E402
from vision import ensure_model  # noqa: E402

ZONE_TAGS = {0: {"center": (0.1, 0.8), "size": 100}, 1: {"center": (0.9, 0.8), "size": 100}}


def new_game():
    random.seed(1)
    g = Game(brain=OpponentBrain(path=None), aspect=0.75)
    g.update(0.0, (0.5, 0.5), ZONE_TAGS)
    g.start()
    return g


def serve_and_arrive(g, t=10.0):
    g.serve(t)
    assert g.state == INCOMING
    return g.ball["t_arrive"], g.ball["to"]


def run_until_judged(g, t_arr, wrist, tags=ZONE_TAGS):
    events = []
    t = t_arr - config.SWING_WINDOW_S
    while g.state == INCOMING:
        events += g.update(t, wrist, tags)
        t += 0.033
    return events


# ------------------------------------------------------------- start -------
def test_start_tag_must_be_held():
    g = Game(brain=OpponentBrain(path=None))
    g.update(0.0, None, {config.TAG_START: {"center": (0.5, 0.5), "size": 100}})
    assert g.state == WAIT_START
    g.update(config.TAG_HOLD_S + 0.01, None, {config.TAG_START: {"center": (0.5, 0.5), "size": 100}})
    assert g.state == READY


def test_zone_comes_from_tags():
    g = new_game()
    assert g.zone == (0.1, 0.9) and g.zone_from_tags
    assert g.zone_points == ((0.1, 0.8), (0.9, 0.8))


def test_court_sidelines_follow_apriltag_centers():
    default_court, _ = court_geometry(((0.1, 0.72), (0.9, 0.72)))
    moved_court, _ = court_geometry(((0.2, 0.65), (0.8, 0.75)))

    assert moved_court[2] == (0.8, 0.75)
    assert moved_court[3] == (0.2, 0.65)
    assert moved_court != default_court


# ------------------------------------------------------------- policy ------
def test_hand_on_ball_plus_swing_is_hit():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    g.add_swing(t_arr + 0.05, "smash")
    events = run_until_judged(g, t_arr, to)
    assert ("hit" in [e for e, _ in events]) and g.streak == 1 and g.record == 1
    assert ("record", 1.0) in events
    assert g.state == OUTGOING


def test_no_swing_is_miss():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    events = run_until_judged(g, t_arr, to)
    assert ("miss", "No swing!") in events and g.streak == 0


def test_swing_too_early_is_miss():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    g.add_swing(t_arr - 1.0, "forehand")
    events = run_until_judged(g, t_arr, to)
    assert events[-1][0] == "miss"


def test_hand_far_away_is_miss():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    g.add_swing(t_arr, "smash")
    far = (to[0] + 0.3 if to[0] < 0.6 else to[0] - 0.3, to[1])
    events = run_until_judged(g, t_arr, far)
    assert events[-1] == ("miss", "Swung, but too far from the ball")


def test_out_of_zone_is_miss():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    moved = {0: {"center": (0.6, 0.8), "size": 100}, 1: {"center": (0.9, 0.8), "size": 100}}
    g.ball["to"] = (0.55, to[1])   # ball near the hand, but hand is left of tag 0
    g.add_swing(t_arr, "smash")
    events = run_until_judged(g, t_arr, (0.55, to[1]), tags=moved)
    assert ("miss", "Out of your zone!") in events


def test_record_keeps_after_miss():
    g = new_game()
    for k in range(3):             # 3 hits in a row
        t_arr, to = serve_and_arrive(g, t=10.0 + 5 * k)
        g.add_swing(t_arr, "smash")
        run_until_judged(g, t_arr, to)
        g.state = READY            # skip the opponent for this test
    t_arr, to = serve_and_arrive(g, t=40.0)
    run_until_judged(g, t_arr, to)  # no swing -> miss
    assert g.streak == 0 and g.record == 3


# ------------------------------------------------------------ opponent ----
def test_opponent_p_controller_moves_toward_ball():
    g = new_game()
    t_arr, to = serve_and_arrive(g)
    g.add_swing(t_arr, "forehand")
    run_until_judged(g, t_arr, to)
    target = g.ball["to"][0]
    err0 = abs(g.opp_x - target)
    t = g.ball["t0"]
    for _ in range(5):
        t += 0.05
        g.update(t, to, ZONE_TAGS)
    assert abs(g.opp_x - target) < err0


def test_q_learning_prefers_rewarded_action():
    b = OpponentBrain(path=None)
    for _ in range(30):
        b.learn(1, 2, 1.0, 1)      # aiming right in state 1 made you miss
        b.learn(1, 0, -0.2, 1)
    assert b.choose(1, epsilon=0.0) == 2


# -------------------------------------------------------------- swings ----
def fake_window(gz_peak, n=40, t0=100.0):
    out = []
    for i in range(n):
        g = gz_peak * max(0.0, 1 - abs(i - 20) / 5)
        out.append((t0 + i * 0.02, 0, 0, 1000, 0, 0, g))
    return out


def test_rule_labels():
    assert rule_based_label(fake_window(2500)[10:26]) == "forehand"
    assert rule_based_label(fake_window(-2500)[10:26]) == "backhand"
    assert rule_based_label(fake_window(6000)[10:26]) == "smash"
    assert rule_based_label(fake_window(200)[10:26]) == "none"


def test_detector_finds_one_swing_at_peak():
    clf = SwingClassifier(path="does_not_exist")
    d = SwingDetector(clf)
    samples = fake_window(3000)
    swings = d.update(samples[:22]) + d.update(samples[22:])   # arrives in pieces
    assert len(swings) == 1
    assert abs(swings[0][0] - samples[20][0]) < 1e-9 and swings[0][1] == "forehand"


def test_fake_paddle_generates_detectable_forehand():
    paddle = FakePaddle()
    paddle.fake_swing()
    detector = SwingDetector(SwingClassifier(path="does_not_exist"))
    swings = detector.update(paddle.get_samples())
    assert len(swings) == 1 and swings[0][1] == "forehand"
    assert swings[0][0] == paddle.get_samples()[config.WINDOW_BEFORE][0]


def test_small_wiggles_are_not_swings():
    d = SwingDetector(SwingClassifier(path="does_not_exist"))
    assert d.update(fake_window(500)) == []


def test_pose_model_download_uses_verified_ca_bundle():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "pose.task")
        with patch("vision.urllib.request.urlopen", return_value=io.BytesIO(b"model")) as urlopen:
            assert ensure_model(path) == path

        with open(path, "rb") as model_file:
            assert model_file.read() == b"model"
        context = urlopen.call_args.kwargs["context"]
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        assert context.get_ca_certs()


def test_daily_record_is_saved_and_resets_for_a_new_day():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "daily_record.json")
        today = date(2026, 10, 12)
        save_daily_record(7, path, today)

        assert load_daily_record(path, today) == 7
        assert load_daily_record(path, date(2026, 10, 13)) == 0


def test_round_screens_draw_clickable_start_and_restart_buttons():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    start_button = draw_round_screen(image.copy(), "instructions", 30, 0, 4)
    result_button = draw_round_screen(image.copy(), "results", 0, 3, 4)

    assert start_button is not None
    assert result_button is not None
    assert start_button == result_button
