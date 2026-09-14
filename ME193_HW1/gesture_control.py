"""
gesture_control.py

Uses MediaPipe Pose to read arm positions from a webcam frame and turns
them into a driving command for the LEGO car.

Gesture map (as specified in the assignment):
    both arms up (vertical)        -> FORWARD
    left arm up, right arm neutral -> TURN_LEFT
    right arm up, left arm neutral -> TURN_RIGHT
    both arms down                 -> STOP
    both arms straight out to the sides (horizontal) -> BACKWARD
    right arm at ~45 degrees up    -> SPEED_UP   (modifier, keeps current direction)
    left arm at ~45 degrees up     -> SLOW_DOWN  (modifier, keeps current direction)

Notes on left/right:
    MediaPipe Pose labels landmarks anatomically (the subject's own left and
    right), regardless of which side of the frame they appear on, as long as
    the raw camera frame is *not* flipped before running the model. That's
    what this module does: it runs pose estimation on the raw frame, and
    only flips the final annotated frame right before it's displayed, purely
    so the on-screen preview feels like a mirror. This keeps "your right arm"
    mapped to `RIGHT_*` landmarks correctly.
"""

import math
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import deque

import cv2
import mediapipe as mp

# MediaPipe removed the old `mp.solutions.pose` API from recent PyPI releases
# (0.10.3x+). This now uses the current "Tasks" API (`mp.tasks.vision.PoseLandmarker`)
# instead. It needs a small model file, which is downloaded automatically the
# first time this runs and cached alongside this script.
_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)
_MIN_VALID_MODEL_BYTES = 500_000  # real model is a few MB; a tiny file means the download failed

# Landmark indices are the same as the old BlazePose topology.
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_LEFT_WRIST, _RIGHT_WRIST = 15, 16

# The standard 33-point BlazePose skeleton connectivity, used only to draw the
# debug overlay. Pulled from mp.tasks.vision.PoseLandmarksConnections when
# that convenience API is available; otherwise this fixed, version-independent
# topology (the same on every MediaPipe release) is used directly.
_FALLBACK_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

try:
    _POSE_CONNECTIONS = [
        (c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    ]
except AttributeError:
    _POSE_CONNECTIONS = _FALLBACK_POSE_CONNECTIONS


def _download(url, dest_path):
    """Download `url` to `dest_path`.

    Python.org's macOS installer ships its own Python build that does NOT
    use the system's trusted root certificates -- until you run the
    'Install Certificates.command' script that comes with it (in
    /Applications/Python 3.x/), any HTTPS request from that Python raises
    ssl.SSLCertVerificationError. Rather than require that one-time manual
    step, this falls back to using the `certifi` package's CA bundle
    (already installed as a dependency of many packages, incl. pip itself),
    and as a last resort disables verification for this one, fixed,
    known-good HTTPS URL.
    """
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        return
    except urllib.error.URLError as exc:
        cert_error = isinstance(exc.reason, ssl.SSLCertVerificationError) or (
            "CERTIFICATE_VERIFY_FAILED" in str(exc)
        )
        if not cert_error:
            raise

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        print(
            "[gesture] Warning: couldn't verify SSL certificates and 'certifi' isn't "
            "installed, so downloading without certificate verification just this once.\n"
            "[gesture] To fix this properly: run 'pip install certifi', or on macOS run "
            "the 'Install Certificates.command' script found in your Python.app/Python 3.x "
            "folder under /Applications."
        )
        ctx = ssl._create_unverified_context()

    with urllib.request.urlopen(url, context=ctx) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())


def _ensure_model_downloaded():
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > _MIN_VALID_MODEL_BYTES:
        return
    print("[gesture] Downloading pose landmarker model (one-time, ~5-30 MB)...")
    _download(_MODEL_URL, _MODEL_PATH)
    if os.path.getsize(_MODEL_PATH) <= _MIN_VALID_MODEL_BYTES:
        size = os.path.getsize(_MODEL_PATH)
        os.remove(_MODEL_PATH)
        raise RuntimeError(
            f"Downloaded model file was only {size} bytes, so the download must have failed "
            "silently (e.g. a firewall/proxy returning an error page instead of the file). "
            f"Try downloading it manually from:\n  {_MODEL_URL}\n"
            f"and save it as:\n  {_MODEL_PATH}"
        )
    print("[gesture] Model downloaded.")

# ---- Gesture / command constants -----------------------------------------
FORWARD = "FORWARD"
BACKWARD = "BACKWARD"
TURN_LEFT = "TURN_LEFT"
TURN_RIGHT = "TURN_RIGHT"
STOP = "STOP"
SPEED_UP = "SPEED_UP"
SLOW_DOWN = "SLOW_DOWN"
HOLD = "HOLD"  # ambiguous pose -> keep doing whatever we were doing

# ---- Arm angle bands (degrees, 90 = straight up, 0 = straight out to the
# side, -90 = straight down) -------------------------------------------------
UP_MIN_DEG = 65        # arm counts as "up" above this
DIAG_MIN_DEG = 25       # arm counts as the ~45 degree diagonal band
DIAG_MAX_DEG = 65
SIDE_MIN_DEG = -20      # arm counts as "out to the side" (horizontal) in this band
SIDE_MAX_DEG = 25
DOWN_MAX_DEG = -20      # arm counts as "down" below this

ARM_UP = "UP"
ARM_DIAG = "DIAG_UP"
ARM_SIDE = "SIDE"
ARM_DOWN = "DOWN"

# How many consecutive frames a pose must be held before we trust it.
# This debounces single-frame jitter/misdetections.
DEBOUNCE_FRAMES = 4


def _angle_from_horizontal(shoulder, wrist):
    """Return the elevation angle (degrees) of the shoulder->wrist vector,
    measured from horizontal. +90 = wrist straight above shoulder (arm up),
    0 = wrist level with shoulder (arm out to the side), -90 = wrist
    straight below shoulder (arm down). Uses image coordinates, where y
    grows downward, hence the sign flip on the vertical component.
    """
    dx = wrist.x - shoulder.x
    dy = shoulder.y - wrist.y  # flip so "up" is positive
    return math.degrees(math.atan2(dy, abs(dx) + 1e-6))


def _classify_arm(angle_deg):
    if angle_deg >= UP_MIN_DEG:
        return ARM_UP
    if DIAG_MIN_DEG <= angle_deg < DIAG_MAX_DEG:
        return ARM_DIAG
    if SIDE_MIN_DEG <= angle_deg < SIDE_MAX_DEG:
        return ARM_SIDE
    return ARM_DOWN


def classify_gesture(left_state, right_state):
    """Combine left/right arm states into a single driving command."""
    if left_state == ARM_DOWN and right_state == ARM_DOWN:
        return STOP
    if left_state == ARM_SIDE and right_state == ARM_SIDE:
        return BACKWARD
    if left_state == ARM_UP and right_state == ARM_UP:
        return FORWARD
    if left_state == ARM_UP and right_state != ARM_UP:
        return TURN_LEFT
    if right_state == ARM_UP and left_state != ARM_UP:
        return TURN_RIGHT
    if right_state == ARM_DIAG and left_state != ARM_UP:
        return SPEED_UP
    if left_state == ARM_DIAG and right_state != ARM_UP:
        return SLOW_DOWN
    return HOLD


class PoseGestureDetector:
    """Wraps MediaPipe's PoseLandmarker task and turns landmarks into
    debounced gesture commands, plus draws a debug overlay."""

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        _ensure_model_downloaded()

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=_MODEL_PATH,
                # Force CPU explicitly: recent mediapipe releases (1.0.x) have a
                # known bug where the pose-detection graph unconditionally tries
                # to initialize a Metal GPU helper on macOS and hard-crashes the
                # process if that service isn't available (common on machines
                # without a strong discrete GPU, e.g. MacBook Air). Pinning to
                # mediapipe==0.10.35 in requirements.txt avoids this entirely;
                # this delegate setting is a second layer of protection.
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._recent = deque(maxlen=DEBOUNCE_FRAMES)
        self._stable_command = STOP

    def close(self):
        self._landmarker.close()

    def _draw_skeleton(self, frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for start, end in _POSE_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)

    def process(self, frame_bgr):
        """Run pose estimation on a raw (un-flipped) BGR frame.

        Returns (annotated_frame, command, debug_info) where:
            annotated_frame: frame with skeleton + text drawn on it, still
                              in the same (un-flipped) orientation as input.
            command: one of the gesture constants, debounced.
            debug_info: dict with left_angle/right_angle/raw command for
                        display or logging.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        debug = {"left_angle": None, "right_angle": None, "raw": HOLD, "detected": False}

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]  # first (only) detected person
            L_SH, L_WR = lm[_LEFT_SHOULDER], lm[_LEFT_WRIST]
            R_SH, R_WR = lm[_RIGHT_SHOULDER], lm[_RIGHT_WRIST]

            left_angle = _angle_from_horizontal(L_SH, L_WR)
            right_angle = _angle_from_horizontal(R_SH, R_WR)
            left_state = _classify_arm(left_angle)
            right_state = _classify_arm(right_angle)
            raw_command = classify_gesture(left_state, right_state)

            debug.update(
                left_angle=left_angle,
                right_angle=right_angle,
                left_state=left_state,
                right_state=right_state,
                raw=raw_command,
                detected=True,
            )

            self._draw_skeleton(frame_bgr, lm)

            self._recent.append(raw_command)
            if len(self._recent) == self._recent.maxlen and len(set(self._recent)) == 1:
                # Every buffered frame agrees -> accept it as the stable command,
                # UNLESS it's a modifier (SPEED_UP/SLOW_DOWN), which should keep
                # firing every time it's held rather than only on first agreement.
                self._stable_command = self._recent[-1]
            elif raw_command in (SPEED_UP, SLOW_DOWN):
                # Modifiers only need to be seen consistently for a couple of
                # frames, not the full debounce window, so speed changes feel
                # responsive.
                if list(self._recent)[-2:] == [raw_command, raw_command]:
                    self._stable_command = raw_command
        else:
            self._recent.clear()

        return frame_bgr, self._stable_command, debug
