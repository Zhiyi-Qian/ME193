"""
Runs the trained step-counting network in plain Python (no PyTorch, no numpy),
so it works inside the UNO Q App. train_steps.py checks that this gives the
same answer as PyTorch.

Network: 3 convolution layers -> 2 fully connected layers -> steps in the
middle 1 s of a 2 s window.
"""
import json
import math

RATE = 50          # samples per second
WIN = 100          # 2 s window
CENTER = 50        # the net predicts steps in the middle 1 s of the window


def make_channels(rows):
    """rows: list of (ax, ay, az, gx, gy, gz). Returns 8 channels:
    the 6 raw signals + acceleration magnitude + rotation magnitude."""
    cols = [list(c) for c in zip(*rows)]
    acc = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[0], cols[1], cols[2])]
    gyr = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[3], cols[4], cols[5])]
    return cols + [acc, gyr]


def _conv_relu(x, W, b):
    """1D convolution with 'same' padding, followed by ReLU."""
    T = len(x[0])
    K = len(W[0][0])
    pad = K // 2
    out = []
    for o in range(len(W)):
        Wo = W[o]
        row = [b[o]] * T
        for i, xi in enumerate(x):
            w = Wo[i]
            for k in range(K):
                wk = w[k]
                shift = k - pad
                lo, hi = max(0, -shift), min(T, T - shift)
                for t in range(lo, hi):
                    row[t] += wk * xi[t + shift]
        out.append([v if v > 0 else 0.0 for v in row])
    return out


def _maxpool2(x):
    return [[max(r[2 * t], r[2 * t + 1]) for t in range(len(r) // 2)] for r in x]


def _linear(v, W, b):
    return [bo + sum(w * x for w, x in zip(Wo, v)) for Wo, bo in zip(W, b)]


class StepNet:
    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.mean, self.std = d["mean"], d["std"]
        self.p = d["params"]

    def predict(self, rows):
        """Expected number of steps in the middle 1 s of a 2 s window of raw rows."""
        x = make_channels(rows)
        x = [[(v - m) / s for v in ch] for ch, m, s in zip(x, self.mean, self.std)]
        p = self.p
        x = _maxpool2(_conv_relu(x, p["conv1.weight"], p["conv1.bias"]))
        x = _maxpool2(_conv_relu(x, p["conv2.weight"], p["conv2.bias"]))
        x = _conv_relu(x, p["conv3.weight"], p["conv3.bias"])
        v = [val for ch in x for val in ch]                       # flatten
        v = [max(0.0, z) for z in _linear(v, p["fc1.weight"], p["fc1.bias"])]
        z = _linear(v, p["fc2.weight"], p["fc2.bias"])[0]
        return z if z > 20 else math.log1p(math.exp(z))           # softplus: never negative
