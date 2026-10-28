from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import sys, os, math, time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ======================= change these =======================
MODE = "record"     # "record" = save training data, "live" = classify
LABEL = "walking"   # record: activity you're recording
                    # live: what you're actually doing, for scoring (None = no scoring)
# ============================================================

RATE, WIN, STEP = 50, 100, 50      # 50 Hz, 2 s windows, predict every 1 s
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
ui = WebUI()

if MODE == "live":
    from gait_rules import model1, model2, MODEL1_AXIS
else:
    os.makedirs("/app/data", exist_ok=True)
    filename = f"/app/data/{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(filename, "w")
    f.write("t,ax,ay,az,gx,gy,gz,label\n")

buf = deque(maxlen=WIN)
batch = []
count = 0
new_samples = 0
total = 0
correct = [0, 0]


def features(rows):
    """Same features as train_models.py, in plain Python."""
    feats = {}
    cols = list(zip(*rows))
    for name, v in zip(CH, cols):
        n = len(v)
        m = sum(v) / n
        feats[f"{name}_mean"] = m
        feats[f"{name}_std"] = math.sqrt(sum((x - m) ** 2 for x in v) / n)
        feats[f"{name}_max"] = max(v)
        feats[f"{name}_min"] = min(v)
    acc = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[0], cols[1], cols[2])]
    gyr = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(cols[3], cols[4], cols[5])]
    n = len(acc)
    am = sum(acc) / n
    feats["acc_mag_std"] = math.sqrt(sum((x - am) ** 2 for x in acc) / (n - 1))
    feats["gyro_mag_mean"] = sum(gyr) / n
    centered = [x - am for x in acc]
    best_k, best_p = 1, -1.0
    for k in range(1, n // 2 + 1):
        re = sum(x * math.cos(2 * math.pi * k * i / n) for i, x in enumerate(centered))
        im = sum(x * math.sin(2 * math.pi * k * i / n) for i, x in enumerate(centered))
        p = re * re + im * im
        if p > best_p:
            best_k, best_p = k, p
    feats["cadence_hz"] = best_k * RATE / n
    axis_vals = cols[CH.index(MODEL1_AXIS)]
    feats[f"peak_{MODEL1_AXIS}"] = max(abs(x) for x in axis_vals)
    return feats


def on_imu(ax, ay, az, gx, gy, gz):
    global count, new_samples, total
    row = (ax, ay, az, gx, gy, gz)

    # stream to the web page 10 times a second (5 samples per message)
    batch.append([round(v, 3) for v in row])
    if len(batch) >= 5:
        ui.send_message("samples", {"rows": batch[:]})
        batch.clear()

    if MODE == "record":
        f.write(f"{time.time():.3f},{ax:.3f},{ay:.3f},{az:.3f},{gx:.2f},{gy:.2f},{gz:.2f},{LABEL}\n")
        count += 1
        if count % 50 == 0:
            f.flush()
            ui.send_message("status", {"mode": "record", "label": LABEL, "count": count})
        if count % 250 == 0:
            print(f"{count} samples saved to {filename}", flush=True)
        return

    # live mode
    buf.append(row)
    new_samples += 1
    if len(buf) < WIN or new_samples < STEP:
        return
    new_samples = 0
    feats = features(list(buf))
    p1, p2 = model1(feats), model2(feats)
    total += 1
    msg = {"mode": "live", "m1": p1, "m2": p2, "truth": LABEL, "windows": total}
    line = f"Model 1: {p1:9s} | Model 2: {p2:9s}"
    if LABEL:
        correct[0] += p1 == LABEL
        correct[1] += p2 == LABEL
        msg["acc1"] = correct[0] / total
        msg["acc2"] = correct[1] / total
        line += f" | accuracy so far: M1 {msg['acc1']:.0%}, M2 {msg['acc2']:.0%} ({total} windows)"
    ui.send_message("prediction", msg)
    print(line, flush=True)


Bridge.provide("imu", on_imu)
App.run()
