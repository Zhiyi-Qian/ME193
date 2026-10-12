# LEGO Ping-Pong 🏓

Virtual ping-pong where the LEGO Double Motor **is** the paddle. Hold it in your hand and swing.

| Class topic | Where it's used |
|---|---|
| Pose (MediaPipe) | `vision.py`: tracks your paddle hand (wrist + elbow) |
| OpenCV / cameras | `vision.py`, `main.py`: camera loop, mirroring, drawing the table |
| AprilTags | Tags 0 and 1 = edges of your play zone, tag 2 = level (distance from camera), tag 3 = start |
| Sound (pyAudio) | `audio_trigger.py`: clap to serve |
| IMU + ML training | `swing.py`, `collect_swings.py`, `train_swing_model.py`: KNN vs decision tree swing classifier |
| Unit tests / ground truth | `tests/test_game.py`, plus a cross-validated confusion matrix from training |
| Proportional control | `game.py`: the opponent paddle is a P controller |
| Q-learning (Q-table) | `game.py`: the opponent learns where you miss most |
| MQTT | `score_mqtt.py`: record streak posted as a float to `ME193/Rogers/YourName` |
| **New: haptics** | `paddle.py`: the motors thump on a sweet-spot hit, buzz lopsided on an edge hit, rattle on a miss, and tick when you leave your zone. The beep pitch climbs with your combo. |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On a Mac, `pyaudio` needs PortAudio first: `brew install portaudio`.

Before you start, edit `config.py`: set `MQTT_BROKER` to the class broker and put your name in `MQTT_TOPIC`.

The first run downloads the MediaPipe pose model (about 6 MB).

**No connection card?** The game connects to the first Double Motor it finds. In a room with other motors, run `python find_motor.py` and copy your motor's address into `MOTOR_MAC` in `config.py`.

## Steps

1. **Test without the motor:** run `python main.py --fake` and press SPACE to swing.
2. **Run the tests:** `python -m pytest -q`
3. **Train your swings:** run `python collect_swings.py`, then `python train_swing_model.py`. Look at the printed peak gyro values and adjust `SWING_GYRO_THRESHOLD` in `config.py` if needed.
4. **Play:** `python main.py`

## How to play

1. Place tag 0 and tag 1 on the left and right edges of where you'll stand. The cyan court outline terminates at the detected centers of the two tags, so move a tag in camera view and its matching court corner/side should move with it. Green circles mark the detected tag centers.
2. Read the rules on the start screen. Click START, press S or Enter, or hold tag 3 in view to begin a 30-second round.
3. Hold tag 2 near the camera for level 1 (easy) or far from it for level 3 (hard). You can also press 1, 2, or 3.
4. Clap to serve, or press C. Move your tracked wrist toward the yellow landing target and swing as the ball arrives. The inner circle is the sweet spot; the outer ring is the hit limit.
5. At the end of the round, the results screen shows the best streak this round and today's best streak. Click PLAY AGAIN or press R or Enter to restart. Today's best is stored locally in `daily_record.json`.

Swinging a forehand on the backhand side counts as an edge hit. A smash sends the ball back faster.
