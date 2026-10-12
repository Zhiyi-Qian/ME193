"""
All the settings for LEGO Ping-Pong in one place.
Change values here instead of digging through the code.
"""

# ---------------------------------------------------------------- MQTT -----
MQTT_BROKER = "broker.hivemq.com"   # <-- change to the broker we use in class
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Rogers/YourName"  # <-- put your name here
MQTT_HEARTBEAT_S = 2.0               # re-publish the record this often

# ------------------------------------------------------------ Paddle -------
# No connection card? Leave these as None and the code connects to the FIRST
# Double Motor it finds. In a classroom full of motors, run find_motor.py once
# and paste your motor's MAC address (or part of its name) here instead.
MOTOR_MAC = None     # e.g. "A1:B2:C3"  (any part of the address works)
MOTOR_NAME = None       # e.g. "Double Motor"
IMU_PERIOD_MS = 20      # IMU sample period (15-1000 ms). 20 ms = 50 Hz

# ------------------------------------------------------- Swing detection ---
# The IMU sends raw integer units. Run collect_swings.py once and look at the
# printed peak gyro values to tune these two numbers for your arm.
SWING_GYRO_THRESHOLD = 1000   # |gyro| above this = "something is happening"
SMASH_GYRO_THRESHOLD = 4000   # used only by the fallback rule (no model yet)
SWING_COOLDOWN_S = 0.35       # ignore new swings right after one
WINDOW_BEFORE = 10            # samples kept before the gyro peak
WINDOW_AFTER = 5              # samples kept after the gyro peak
MODEL_PATH = "swing_model.joblib"
DATA_PATH = "swing_data.jsonl"

# --------------------------------------------------------- AprilTags -------
TAG_LEFT_ZONE = 0
TAG_RIGHT_ZONE = 1
TAG_LEVEL = 2
TAG_START = 3
TAG_HOLD_S = 0.6              # hold the START tag this long to begin

# Level from how big tag 2 looks (its side length in pixels).
# Close to the camera (big) = level 1 (easy). Far (small) = level 3 (hard).
LEVEL_SIZE_THRESHOLDS = (160, 90)   # >160 px -> 1, 90-160 -> 2, <90 -> 3

# Per level: ball travel time (s), opponent P-gain (higher = opponent reaches
# more balls), Q-learning exploration rate (lower = opponent exploits your
# weak side more often)
LEVELS = {
    1: dict(travel_time=1.8, opp_gain=2.5, epsilon=0.6),
    2: dict(travel_time=1.3, opp_gain=4.0, epsilon=0.3),
    3: dict(travel_time=0.9, opp_gain=7.0, epsilon=0.1),
}
OPP_MAX_SPEED = 1.2     # opponent paddle speed limit (screen widths per second)
OPP_REACH = 0.07        # opponent returns the ball if within this distance

# ------------------------------------------------------------ Pose ---------
HAND = "right"                 # which hand holds the motor: "right" or "left"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

# ------------------------------------------------------------ Game ---------
ROUND_DURATION_S = 30
DAILY_RECORD_PATH = "daily_record.json"
HIT_RADIUS = 0.13       # wrist must be this close to the ball (fraction of width)
SWEET_RADIUS = 0.06     # this close = sweet spot
SWING_WINDOW_S = 0.30   # swing must happen within +/- this of ball arrival
BALL_ARRIVAL_Y = 0.55   # where the ball arrives vertically (fraction of height)
Q_TABLE_PATH = "q_table.json"

# ------------------------------------------------------------ Audio --------
AUDIO_ENABLED = True
CLAP_FACTOR = 6.0       # a clap must be this many times louder than the room
CLAP_MIN_RMS = 0.05     # ...and at least this loud (0-1 scale)
