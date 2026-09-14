# Gesture-Controlled LEGO Car

Control a LEGO Education Computer Science & AI kit car (Double Motor hub)
by moving your arms in front of a webcam. Built with **Python**,
**MediaPipe Pose**, and LEGO's official `legoeducation` PyPI package.

## Gesture map

| Gesture                                   | Action              |
|--------------------------------------------|---------------------|
| Both arms straight up                      | Drive forward       |
| Left arm up (right arm not up)             | Turn left           |
| Right arm up (left arm not up)             | Turn right          |
| Both arms down                             | Stop                |
| Both arms straight out to the sides        | Drive backward      |
| Right arm at ~45° up                       | Speed up            |
| Left arm at ~45° up                        | Slow down           |

Speed up / slow down adjust the current driving speed by a small step
each time the gesture is held, without changing direction — hold the arm
at 45° to keep accelerating/decelerating up to the min/max speed limits.

## How it works

- `gesture_control.py` — runs MediaPipe Pose on each webcam frame, computes
  the elevation angle of each arm (shoulder → wrist), classifies each arm as
  UP / DIAG_UP (~45°) / SIDE / DOWN, and combines both arms into one of the
  commands above. A short debounce window filters out single-frame jitter.
- `car_controller.py` — wraps LEGO's `legoeducation.DoubleMotor` class and
  turns commands into `movement_move_tank(left_speed, right_speed)` calls.
  Turning is done by slowing (not fully stopping) the inside wheel.
- `main.py` — captures webcam frames, ties the two together, and shows a
  debug window with the detected gesture, arm angles, and current speed.

## Setup

1. Charge and power on the LEGO Double Motor hub and make sure it's
   broadcasting over Bluetooth.
2. Install dependencies (Python 3.11+ required by `legoeducation`):
   ```
   pip install -r requirements.txt
   ```
3. Run it:
   ```
   python main.py
   ```
   The program scans for and connects to the first Double Motor hub it
   finds, then opens a webcam window.

   Don't have the hub in front of you / just want to test the gesture
   detection? Run in simulate mode — it prints the commands instead of
   sending them over Bluetooth:
   ```
   python main.py --simulate
   ```

   Use a different webcam:
   ```
   python main.py --camera 1
   ```

Press **q** or **Esc** in the video window to quit — the car is
automatically stopped and disconnected on exit.

## Notes / tuning

- This uses MediaPipe's current **Tasks API** (`mediapipe.tasks.vision.PoseLandmarker`),
  not the older `mp.solutions.pose` API — recent `mediapipe` releases (0.10.3x+
  on PyPI) dropped `mp.solutions` entirely, so `mp.solutions.pose` now raises
  `AttributeError: module 'mediapipe' has no attribute 'solutions'`. The first
  time you run `main.py`, it automatically downloads a small pose model file
  (`pose_landmarker_lite.task`, a few MB) into this folder and reuses it after
  that — you need an internet connection the first time only.
- MediaPipe labels landmarks based on the person's actual left/right arm
  (not mirrored), as long as it's run on the raw camera frame — which is
  what this project does. The preview window is flipped only *after*
  detection, purely so it feels like looking in a mirror.
- Angle thresholds (what counts as "up", "~45°", "to the side", "down")
  live at the top of `gesture_control.py` if they need tuning for your
  camera angle or arm length.
- Turn sharpness, base/min/max speed, and how fast speed-up/slow-down
  ramp are all constants at the top of `car_controller.py`.
