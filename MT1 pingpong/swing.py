"""
Swing recognition from the Double Motor IMU.

1. SwingDetector watches |gyro|. When it crosses SWING_GYRO_THRESHOLD it finds
   the peak and cuts out a fixed window of samples around it.
2. features() turns that window into a short list of numbers.
3. A trained classifier (k-nearest neighbours or decision tree, see
   train_swing_model.py) labels it: forehand / backhand / smash / none.
   If no model has been trained yet, a simple hand-written rule is used.
"""

import math
import os

import numpy as np

import config

LABELS = ["forehand", "backhand", "smash", "none"]
WINDOW_LEN = config.WINDOW_BEFORE + 1 + config.WINDOW_AFTER


def gyro_mag(s):
    return math.sqrt(s[4] ** 2 + s[5] ** 2 + s[6] ** 2)


def features(window):
    """
    window: list of (t, ax, ay, az, gx, gy, gz) samples, length WINDOW_LEN.
    Returns: 1-D numpy array of features.
      for each of the 6 axes: max, min, mean, std   (24 numbers)
      + peak |gyro| + peak |accel|                  (2 numbers)
    """
    a = np.array([s[1:7] for s in window], dtype=float)  # shape (N, 6)
    f = [a.max(0), a.min(0), a.mean(0), a.std(0)]
    gmag = np.linalg.norm(a[:, 3:6], axis=1).max()
    amag = np.linalg.norm(a[:, 0:3], axis=1).max()
    return np.concatenate(f + [[gmag, amag]])


def rule_based_label(window):
    """Fallback when there's no trained model yet."""
    peak = max(window, key=gyro_mag)
    if gyro_mag(peak) < config.SWING_GYRO_THRESHOLD:
        return "none"
    if gyro_mag(peak) > config.SMASH_GYRO_THRESHOLD:
        return "smash"
    # Forehand and backhand rotate the wrist in opposite directions,
    # so the sign of the yaw (z) rotation tells them apart.
    return "forehand" if peak[6] > 0 else "backhand"


class SwingClassifier:
    def __init__(self, path=config.MODEL_PATH):
        self.model = None
        if os.path.exists(path):
            import joblib
            self.model = joblib.load(path)
            print(f"Loaded swing model from {path}")
        else:
            print("No swing model yet - using the simple rule. "
                  "Run collect_swings.py and train_swing_model.py to train one.")

    def predict(self, window):
        if self.model is None:
            return rule_based_label(window)
        return str(self.model.predict([features(window)])[0])


class SwingDetector:
    """Feed it new IMU samples; it returns finished swings as (time, label)."""

    def __init__(self, classifier=None):
        self.clf = classifier or SwingClassifier()
        self.buf = []            # recent samples
        self.last_t = 0.0        # newest sample time already seen
        self.cooldown_until = 0.0

    def update(self, new_samples):
        swings = []
        for s in new_samples:
            if s[0] <= self.last_t:
                continue
            self.last_t = s[0]
            self.buf.append(s)
        self.buf = self.buf[-(WINDOW_LEN * 4):]

        # Look for a peak that has enough samples before AND after it.
        while True:
            i = self._find_peak()
            if i is None:
                break
            window = self.buf[i - config.WINDOW_BEFORE: i + config.WINDOW_AFTER + 1]
            t_peak = self.buf[i][0]
            label = self.clf.predict(window)
            self.cooldown_until = t_peak + config.SWING_COOLDOWN_S
            self.buf = self.buf[i + 1:]
            if label != "none":
                swings.append((t_peak, label))
        return swings

    def _find_peak(self):
        n = len(self.buf)
        for i in range(config.WINDOW_BEFORE, n - config.WINDOW_AFTER):
            s = self.buf[i]
            if s[0] < self.cooldown_until:
                continue
            g = gyro_mag(s)
            if g < config.SWING_GYRO_THRESHOLD:
                continue
            # it's a peak if nothing in the window after it is bigger
            after = self.buf[i + 1: i + config.WINDOW_AFTER + 1]
            if all(gyro_mag(x) <= g for x in after):
                return i
        return None
