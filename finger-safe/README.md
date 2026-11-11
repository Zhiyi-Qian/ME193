# Finger Safe

A lock that opens when you show the right three numbers with your fingers.

You press a button on the safe, its light turns on, and you hold up your code one number at a time inside a pink frame under a webcam. A convolutional neural network, built from scratch in PyTorch, counts your fingers. If the three numbers match the code, a DC motor slides the latch open.

Built for ME193 Midterm 2 on an Arduino UNO Q, by Zhiyi.

## What's in this repo

| File | What it does |
|---|---|
| `collect_data.py` | Records training photos from the webcam into `data/0` … `data/10`, 200 at a time, hands-free |
| `train.py` | Defines the CNN, trains it, and saves `finger_model.pth` plus training-curve and confusion-matrix plots |
| `live_test.py` | Shows the model's live guess for every frame, to check accuracy before building the lock |
| `lock.py` | Simple version of the lock in an OpenCV window |
| `lock_web.py` | Full version of the lock: runs the camera, model and lock policy, and serves the web interface |
| `lock_ui.html` | The web interface: live camera, step-by-step instructions, code dial, and the "What the network sees" view |
| `unoq_app/` | The Arduino App Lab app for the UNO Q: button, LED and motor control |
| `requirements.txt` | Python libraries needed on the laptop |

Not included: the training photos (`data/`, ignored by git) and `roi.json`, which `collect_data.py` creates the first time you mark the detection area. `finger_model.pth` is created by `train.py`.

## How it works

### The neural network

The network takes the part of the camera image inside the pink frame, shrunk to 96 × 96 pixels, and outputs a score for each of 11 answers: 0–9 fingers, or no hand.

```
input 96×96×3
 → conv 5×5, 16 filters → ReLU → max pool   (48×48)   finds edges and skin/table borders
 → conv 3×3, 32 filters → ReLU → max pool   (24×24)   combines edges into finger-shaped pieces
 → conv 3×3, 64 filters → ReLU → max pool   (12×12)   responds to whole fingers and their positions
 → flatten → dropout 0.3 → linear → 11 scores
```

It has about 120,000 weights. Training uses cross-entropy loss (we want a category, not a number) and the Adam optimizer: each batch runs forward, the loss is backpropagated to get the gradient for every weight, and gradient descent nudges the weights downhill.

To make the most of the photos, training randomly rotates (±15°), shifts, zooms, flips and changes the brightness of each image. Validation uses the *last* 20% of each class's recording rather than random photos, because burst photos next to each other are nearly identical and would make the score look better than it is.

Results: the 10-class model reached 97.88% validation accuracy after 50 epochs. *(Add your 11-class result here.)*

### The lock policy

```
IDLE      button pressed ───────────────► READING (light on)
READING   show a fist or empty frame, then a number
          a number counts only when 8 frames in a row agree at ≥80% confidence
          repeat for each number
          right code ───────────────────► OPEN (motor opens latch)
          wrong code ───────────────────► IDLE (light flashes)
          15 s with no new number ──────► IDLE
OPEN      button pressed ───────────────► IDLE (motor closes latch)
```

Requiring a fist or empty frame between numbers means holding one number for a long time can't enter it twice. Requiring several agreeing frames filters out single-frame mistakes.

### What the network sees

The web page shows the image travelling through the network: the 96 × 96 input, the 9 most active filters of each convolution layer, and a Grad-CAM heatmap. Grad-CAM takes the gradient of the winning score with respect to the last convolution layer, uses it to weight that layer's filters, and overlays the result on the hand, so red shows where the network looked to make its decision.

## Hardware

- Arduino UNO Q, plus the Arduino USB-C hub
- USB webcam on a fixed stand, pointing straight down at a pink tape frame
- Pushbutton, one LED with a 220 Ω resistor
- DC motor driving the latch, through a Cytron Maker Drive
- 4×AA battery box for the motor

The same webcam, stand height and frame must be used for training and for the lock, because the model learns what images from that exact setup look like.

### Wiring

| From | To |
|---|---|
| Pushbutton | UNO Q pin 2 and GND |
| LED (+ 220 Ω resistor) | UNO Q pin 4 → resistor → LED → GND |
| Maker Drive header M1A | UNO Q pin 9 |
| Maker Drive header M1B | UNO Q pin 10 |
| Maker Drive header GND | UNO Q GND |
| Maker Drive VB+ / VB− | Battery box red / black |
| Maker Drive screw terminals M1A / M1B | The two motor wires |

The motor runs at reduced speed (PWM 120 of 255) for 3 seconds to open or close. Change `MOTOR_SPEED` and `MOTOR_MS` at the top of `unoq_app/sketch/sketch.ino` to suit your latch. If it moves the wrong way, swap the two calls in `moveLatch()` or swap the motor wires.

## Setup and use

### 1. Laptop

```
pip3 install -r requirements.txt
```

Plug the webcam into the laptop (through the hub or an adapter) and find its number by trying `--camera 0`, `1` or `2` below.

### 2. Collect data and train

```
python3 collect_data.py --camera 0 --person yourname            # all classes, 200 photos each
python3 collect_data.py --camera 0 --person yourname --labels 3  # just one class
python3 train.py
python3 live_test.py --camera 0
```

The first run of `collect_data.py` asks you to drag a box around the inside of the tape frame. For each class, move your hand slowly while it records: different finger combinations, slight tilts, different spots in the frame. For class 10 (no hand), keep the frame empty but vary the lighting and shadows.

### 3. UNO Q app

1. Connect the UNO Q directly to the laptop with a USB-C data cable and open Arduino App Lab.
2. Create a new app. Paste `unoq_app/sketch/sketch.ino` into the sketch and `unoq_app/python/main.py` into the Python file, and put `unoq_app/assets/index.html` in the app's `assets` folder.
3. Add the **Web UI** brick.
4. Move the latch to closed by hand, then click Run.
5. Open `http://<UNO-Q-IP>:7000` and use the test page to check the light, motor and button.

The UNO Q and the laptop must be on the same Wi-Fi. If the test page won't load on a school network, use a phone hotspot for both.

### 4. Run the lock

```
python3 lock_web.py --camera 0 --code 3 1 5 --board http://<UNO-Q-IP>:7000
```

Open `http://localhost:8000`. Press the button on the safe (or **Wake the safe** on the page), then follow the instructions. Useful options: `--show-digits` shows entered numbers instead of dots, `--viz-every 6` updates the network view less often, and `--no-viz` turns it off.

## Limitations

- Lighting changes, such as a lamp or sunlight on the frame, make images look different from the training photos.
- It's trained mostly on one person's hands, so other people's hands may be read less reliably.
- Anyone watching can see the code as it's entered.
- The motor moves by time, not position, so a weak battery or extra friction can stop the latch short.
- The camera, stand and frame can't move after training without collecting new data.
