# HW7 Step Counter

A wearable step counter built on my HW6 gait identifier. An Arduino UNO Q reads an MPU-6050 IMU strapped to my ankle, and a small **1D convolutional neural network trained in PyTorch** counts steps live from all six sensor outputs. The count shows on a phone dashboard served by the board.

Demo video: *(add link)*  
Neural net animation: *(add link)*  
Notion page: *(add link)*

## Hardware

| Part | Notes |
|---|---|
| Arduino UNO Q | Microcontroller reads the sensor; on-board Linux runs the neural net and web server |
| MPU-6050 | 3-axis accelerometer + 3-axis gyroscope, I2C address `0x68` |
| 4 female-to-male jumper wires | |
| USB-C power bank + ankle strap | Makes it wearable |

| MPU-6050 | UNO Q |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | SDA (header next to AREF) |
| SCL | SCL (header next to AREF) |

## How it works

```
MPU-6050 ──I2C──► STM32 sketch (50 Hz) ──Bridge──► Python on Linux ──WebSocket──► phone dashboard
                  ±8 g, ±1000 °/s,                 every 1 s: last 2 s of data
                  gyro calibration                 → neural net → steps in that second
                                                   → running total
```

1. **Sensing (`app/sketch/sketch.ino`).** The microcontroller reads all six IMU outputs 50 times per second, subtracts the gyroscope's resting offset (measured during the first second), and sends each reading to Python.
2. **Inputs.** Every second, the last 2 seconds of data (100 samples) become the network's input: the six raw signals plus acceleration magnitude and rotation magnitude, for **8 channels × 100 samples**.
3. **Neural net.** The network predicts how many steps happened in the **middle 1 second** of that window. The extra half second on each side gives it context, and because the window slides by exactly 1 second, the middle pieces tile the recording with no gaps or overlaps. Adding up the predictions gives the total count.
4. **On the board.** PyTorch can't run inside the UNO Q App, so the trained weights are exported to `step_model.json`, and `step_net.py` runs the same math in plain Python. The training script checks that both versions give the same answer.

### The network

| Layer | Output shape | What it does |
|---|---|---|
| Input | 8 × 100 | 2 s of sensor data, normalized |
| Conv1d (12 filters, width 5) + ReLU + MaxPool | 12 × 50 | detects short motion shapes, like a heel strike or leg swing |
| Conv1d (16 filters, width 5) + ReLU + MaxPool | 16 × 25 | combines those shapes into longer patterns |
| Conv1d (16 filters, width 5) + ReLU | 16 × 25 | refines the patterns |
| Flatten → Linear (16) + ReLU | 16 | combines all patterns with their timing |
| Linear (1) + Softplus | 1 | steps in the middle second (never negative) |

9,197 trainable weights in total.

### Training

- **Labels.** Each recording's file name holds the true step count, which I counted out loud while walking (for example `walk_100_...` = 100 steps, counting every footfall of both feet). The script finds when I was moving (smoothed rotation above a threshold) and spreads the counted steps evenly over that time. Each window's label is the number of steps in its middle second.
- **Learning.** The network starts with random weights. For each batch of 64 windows it makes predictions, measures the mean squared error against the labels, and **back-propagation** computes how much each weight contributed to that error. The Adam optimizer then nudges every weight slightly in the direction that lowers it. This repeats for 40 passes (epochs) through the data, with small random noise added to the inputs so the network doesn't memorize exact recordings.
- **Data.** 10 recordings: 3 normal walks (100 steps), 2 slow walks (60), 2 fast walks (100), 2 stop-and-go walks (80, with pauses), and 1 minute of standing (0).

![Loss curve](plots/loss_curve.png)

## Uncertainty analysis: how good is it?

I tested with **leave-one-recording-out cross-validation**. For each of the 10 recordings, a fresh network was trained on the other 9 and then counted the steps in the one it never saw. So every number below is a fair test on unseen data.

| Recording | True steps | Predicted | Error |
|---|---|---|---|
| walk 1 | 100 | 98.7 | −1.3% |
| walk 2 | 100 | 96.7 | −3.3% |
| walk 3 | 100 | 93.9 | −6.1% |
| stop-and-go 1 | 80 | 80.1 | +0.1% |
| stop-and-go 2 | 80 | 84.5 | +5.6% |
| slow walk 1 | 60 | 68.5 | +14.2% |
| slow walk 2 | 60 | 67.1 | +11.8% |
| fast walk 1 | 100 | 90.9 | −9.1% |
| fast walk 2 | 100 | 74.7 | −25.3% |
| standing (60 s) | 0 | 21.4 | 21 false steps |

**Summary over the 9 walking recordings:**

- Mean absolute error: **7.3 steps** per recording
- Mean error: **−1.5%**, with a standard deviation of **11.9%**
- 95% confidence interval of the mean error: **−1.5% ± 9.1%**
- Across all 780 walking steps, it counted 755, **3.2% under**

![Predicted vs true steps](plots/predicted_vs_true.png)

**What this means.** At a normal pace, and when stopping and starting, the counter is accurate to within about 6%. It does worse at the edges of what it was trained on: it overcounts slow walking by about 13%, undercounts the fastest walk (138 steps/min) by 25%, and counts false steps while standing and shifting weight. These failures have one cause: a neural network mostly **interpolates** between examples it has seen, and it can't reliably **extrapolate** beyond them. When the fastest walk was held out, the fastest one left in training was only 120 steps/min. When the only standing recording was held out, the training data had no "moving but not walking" examples at all. The final network was trained on all 10 recordings, so it covers this range better, but more recordings at very slow and very fast paces, and more standing data, would be the clearest way to improve it.

**Limitations.** It's trained on walking only (not running or biking), on one person, with the sensor on one ankle. That sensor clearly feels only its own leg's footfalls, so the network has to learn to count the other leg's steps from the rhythm.

## Repository layout

```
HW7_StepCounter/
├── app/                        # the Arduino App Lab app
│   ├── sketch/sketch.ino       # reads and calibrates the IMU at 50 Hz
│   ├── python/main.py          # live step counter + dashboard messages
│   ├── python/record_main.py   # recording mode used to collect training data
│   ├── python/step_net.py      # the trained network in plain Python
│   ├── python/step_model.json  # trained weights
│   └── assets/index.html       # live dashboard
├── training/
│   ├── train_steps.py          # labels, PyTorch training, cross-validation, plots, export
│   └── step_net.py
├── steps/                      # labeled recordings: t, ax, ay, az, gx, gy, gz, label
└── plots/                      # loss curve, predicted vs true
```

To rebuild the app, create an App in Arduino App Lab, add the **WebUI - HTML** brick (this creates `app.yaml` and `assets/libs/` with the Socket.IO client), then copy in the files from `app/`.

## How to run it

**Record training data:** use `record_main.py` as the App's `main.py`. Set `LABEL` to the activity and the exact number of steps (for example `walk_100`), click Run, stand still for 3 seconds, walk while counting every step out loud, stand still for 3 seconds, and stop.

**Train (on a computer, not the board):**

```
pip install torch numpy pandas matplotlib scipy
cd training
python3 train_steps.py
```

It expects the CSVs in a `steps/` folder next to it, and it writes `step_model.json`, `loo_results.csv`, and the plots.

**Run live:** copy `step_net.py` and `step_model.json` into the App's `python/` folder, use `main.py` from `app/python/`, click Run, stay still for 1 second, and open `http://<UNO-Q-IP>:7000`. Press **Reset** before each walk.

## Notes from setting up

- **Wrong clock.** A new UNO Q may think it's still the date its software was built, which breaks library downloads with a certificate error. Fix it with `sudo date -u -s "YYYY-MM-DD HH:MM:SS"`.
- **Campus Wi-Fi.** Networks with device registration (like Tufts Wi-Fi) block the board until it's registered. A phone hotspot works for downloads.
- **Avoid `apt upgrade` on the board.** Running system updates on my first board broke its Wi-Fi driver after the next reboot. Training on a laptop avoids needing any extra software on the board.
