# HW6 Gait Identifier

A wearable that tells whether you're **standing, walking, running, or biking**, built from an Arduino UNO Q and an MPU-6050 IMU strapped to the ankle. It compares two non-neural-net models side by side and shows the result live on a phone dashboard that changes color with the current activity.

- **Model 1** uses only the peak acceleration along one axis.
- **Model 2** uses features from all six IMU outputs (3-axis accelerometer + 3-axis gyroscope).

Demo video: *(add link)*  
Notion page: *(add link)*

## Hardware

| Part | Notes |
|---|---|
| Arduino UNO Q | Microcontroller reads the sensor; on-board Linux runs the Python classifier and web server |
| MPU-6050 | 6-axis IMU (accelerometer + gyroscope), I2C address `0x68` |
| 4 female-to-male jumper wires | |
| USB-C power bank | Makes the wearable untethered |
| Ankle strap | Sensor is always mounted in the same orientation |

**Wiring**

| MPU-6050 | UNO Q |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | SDA (header next to AREF) |
| SCL | SCL (header next to AREF) |

## How it works

```
MPU-6050 ──I2C──► STM32 sketch (50 Hz) ──Bridge──► Python on Linux ──WebSocket──► phone dashboard
                  read + calibrate                  2 s windows → features
                                                    → Model 1 & Model 2
```

1. **Sensing (`app/sketch/sketch.ino`).** The microcontroller reads the IMU 50 times per second. At startup it measures the gyroscope's resting offset for one second and subtracts it from every reading. The sensor ranges are set to ±8 g and ±1000 °/s, because the default ±2 g / ±250 °/s range saturated during running at the ankle.
2. **Windowing.** Python on the board groups the stream into 2-second windows (100 samples) and makes a new prediction every second.
3. **Features.** Each window is summarized into a few numbers, and the two models differ only in which numbers they see.
4. **Classification.** Both models are decision trees trained with scikit-learn, then exported to plain `if/else` Python (`gait_rules.py`) so the board needs no ML libraries at runtime.
5. **Dashboard (`app/assets/index.html`).** A web page served from the board on port 7000. The screen floods with the predicted activity's color, shows both models' guesses and running accuracy, a 60-second prediction history, and live acceleration and rotation plots.

## The two models

### Model 1: peak acceleration on one axis

One feature per window: the largest absolute acceleration on the **z-axis** (picked automatically as the most separating axis). A depth-3 decision tree turns that single number into thresholds:

| Peak \|a<sub>z</sub>\| in a 2 s window | Prediction |
|---|---|
| below 0.60 g | standing |
| 0.60 – 1.89 g | walking |
| 1.89 – 4.25 g | biking |
| above 4.25 g | running |

![Model 1 thresholds](plots/model1_thresholds.png)

### Model 2: all IMU outputs

Twenty-eight features per window: the mean, standard deviation, max, and min of all six channels, plus overall acceleration variability, mean rotation speed, and dominant frequency (step or pedal cadence). A depth-4 decision tree with at least 5 windows per leaf picked these questions:

1. **`gy_max` ≤ 22.7 °/s?** The shin is barely rotating → **standing**.
2. **`ax_max` ≤ 0.25 g?** A smooth motion without sharp jolts → walking or biking, decided by **`ax_mean`** (leg tilt).
3. Otherwise, a jolty motion → running or biking, decided by **`ay_mean`** (leg tilt).

![Model 2 decision tree](plots/model2_tree.png)

## Results

Each recording was split in time: the first 70% trained the models and the last 30% tested them, so test windows were never seen during training.

| | Model 1 | Model 2 |
|---|---|---|
| Test accuracy (89 held-out windows) | 80.9% | **95.5%** |

![Confusion matrices](plots/confusion_matrices.png)

**Live accuracy** (wearing the device, about one minute per activity):

| Activity | Model 1 | Model 2 |
|---|---|---|
| Standing | 100% | 100% |
| Walking | *(fill in)* | *(fill in)* |
| Running | *(fill in)* | *(fill in)* |
| Biking | *(fill in)* | *(fill in)* |

### Which model works better, and why

**Model 2 is better overall.** Model 1 judges an activity only by how hard the foot hits, so every activity gets a single band on one number line, and a hard walking step or a soft running stride lands in the wrong band. Model 2 can also see **how the leg rotates** (gyroscope) and **how it is tilted** (the direction of gravity in the accelerometer mean). Pedaling holds the shin at a different angle than walking or running, which a single acceleration peak cannot capture.

Model 2's advantage has a cost: tilt features depend on how the strap sits, so the sensor has to be mounted the same way each time. Model 1 is simpler and less sensitive to mounting.

### What changed between versions

- **Version 1** trained on about 45 s per activity at the default sensor range. Model 2 scored 100% offline but failed live, mixing up running and biking: it had overfit to quirks of a small dataset, and running impacts were clipped at 2 g.
- **Version 2** raised the sensor range to ±8 g / ±1000 °/s, re-recorded about 1–2 minutes per activity, and required at least 5 windows per tree leaf. Offline accuracy became more realistic (95.5%), and running impacts above 4 g became measurable.

## Repository layout

```
HW6_GaitIdentifier/
├── app/                      # the Arduino App Lab app
│   ├── sketch/sketch.ino     # reads and calibrates the IMU, sends samples over Bridge
│   ├── python/main.py        # record mode (saves CSV) or live mode (classifies + dashboard)
│   ├── python/gait_rules.py  # exported decision trees as plain Python
│   └── assets/index.html     # live dashboard
├── training/
│   ├── train_models.py       # windows, features, trains both models, saves plots
│   ├── export_rules.py       # converts trained trees into gait_rules.py
│   └── gait_rules.py
├── data/                     # labeled recordings: t, ax, ay, az, gx, gy, gz, label
└── plots/                    # threshold histogram, decision tree, confusion matrices
```

To rebuild the app, create a new App in Arduino App Lab, add the **WebUI - HTML** brick (this also creates `app.yaml` and the `assets/libs/` folder with the Socket.IO client), then copy in the files from `app/`.

## How to run it

**Record training data**

1. In `app/python/main.py`, set `MODE = "record"` and `LABEL` to the activity.
2. Strap the sensor on, click **Run** in App Lab, and stay still for one second while the gyroscope calibrates.
3. Do the activity for 2–3 minutes. The CSV is saved in the app's `data/` folder.

**Train**

```
sudo apt install -y python3-pandas python3-sklearn python3-matplotlib python3-joblib
cd training
MPLBACKEND=Agg python3 train_models.py
python3 export_rules.py
cp gait_rules.py ../app/python/
```

`train_models.py` expects the CSVs in a `data/` folder next to it.

**Run live**

1. Set `MODE = "live"`, and set `LABEL` to the true activity for scoring or `None` for a free demo.
2. Click **Run** and stay still for one second.
3. Open `http://<UNO-Q-IP>:7000` on a phone on the same Wi-Fi.
