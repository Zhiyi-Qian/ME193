from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import sys, os, time, threading, queue
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step_net import StepNet, WIN, CENTER

net = StepNet(os.path.join(HERE, "step_model.json"))
ui = WebUI()

buf = deque(maxlen=WIN)     # the last 2 s of sensor rows
batch = []                  # rows waiting to be sent to the plots
new_samples = 0
jobs = queue.Queue()
lock = threading.Lock()
state = {"total": 0.0, "recent": deque(maxlen=10)}   # recent = steps found in each of the last 10 s


def payload(last=0.0, ms=0):
    recent = list(state["recent"])
    cadence = sum(recent) / len(recent) * 60 if recent else 0.0
    return {"total": round(state["total"]), "exact": round(state["total"], 1),
            "cadence": round(cadence), "last": round(last, 2), "ms": ms}


def reset():
    with lock:
        state["total"] = 0.0
        state["recent"].clear()
        msg = payload()
    msg["reset"] = True
    ui.send_message("steps", msg)
    print("Counter reset", flush=True)
    return {"ok": True}


ui.expose_api("GET", "/reset", reset)


def worker():
    """Runs the neural net once per second, off the sensor thread so no samples are missed."""
    while True:
        rows = jobs.get()
        t0 = time.time()
        steps = net.predict(rows)            # steps in the middle 1 s of this 2 s window
        ms = int((time.time() - t0) * 1000)
        with lock:
            state["total"] += steps
            state["recent"].append(steps)
            msg = payload(steps, ms)
        ui.send_message("steps", msg)
        print(f"total {msg['exact']:7.1f} steps | {msg['cadence']:3d} steps/min | "
              f"this second {steps:4.2f} | net took {ms} ms", flush=True)


threading.Thread(target=worker, daemon=True).start()


def on_imu(ax, ay, az, gx, gy, gz):
    global new_samples
    row = (ax, ay, az, gx, gy, gz)
    buf.append(row)

    batch.append([round(v, 3) for v in row])
    if len(batch) >= 5:
        ui.send_message("samples", {"rows": batch[:]})
        batch.clear()

    new_samples += 1
    if len(buf) == WIN and new_samples >= CENTER:   # every 1 s, once 2 s of data exist
        new_samples = 0
        jobs.put(list(buf))


Bridge.provide("imu", on_imu)
App.run()
