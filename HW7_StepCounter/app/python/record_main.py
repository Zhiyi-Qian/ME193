from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import os, time

# ======================= change this =======================
LABEL = "walk_100"   # activity + the exact number of steps you will take, e.g. "run_100", "stand_0"
# ===========================================================

ui = WebUI()
os.makedirs("/app/steps", exist_ok=True)
filename = f"/app/steps/{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
f = open(filename, "w")
f.write("t,ax,ay,az,gx,gy,gz,label\n")

batch = []
count = 0


def on_imu(ax, ay, az, gx, gy, gz):
    global count
    f.write(f"{time.time():.3f},{ax:.3f},{ay:.3f},{az:.3f},{gx:.2f},{gy:.2f},{gz:.2f},{LABEL}\n")
    count += 1

    # stream to the dashboard 10 times a second
    batch.append([round(v, 3) for v in (ax, ay, az, gx, gy, gz)])
    if len(batch) >= 5:
        ui.send_message("samples", {"rows": batch[:]})
        batch.clear()

    if count % 50 == 0:
        f.flush()
        ui.send_message("status", {"mode": "record", "label": LABEL, "count": count})
    if count % 250 == 0:
        print(f"{count} samples saved to {filename}", flush=True)


Bridge.provide("imu", on_imu)
App.run()
