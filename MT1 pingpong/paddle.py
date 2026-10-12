"""
The LEGO Double Motor used as a hand-held paddle.

- Streams the IMU (accelerometer + gyroscope) into a buffer with timestamps.
- Gives "haptic" feedback by pulsing the motors, plus light and beep effects.
  Motor commands run on their own thread so the game never waits on Bluetooth.

FakePaddle has the same interface and lets you test without the hardware
(press SPACE in the game window to "swing").
"""

import queue
import threading
import time
from collections import deque

import config

try:
    import legoeducation as le
except ImportError:  # lets the rest of the project import without the library
    le = None


class Paddle:
    def __init__(self):
        if le is None:
            raise RuntimeError("pip install legoeducation")
        self.motor = le.DoubleMotor()
        self.samples = deque(maxlen=500)   # (t, ax, ay, az, gx, gy, gz)
        self._lock = threading.Lock()
        self._cmds = queue.Queue()
        self._running = False

    # ------------------------------------------------------------ connect --
    def connect(self):
        print("Searching for the Double Motor (press its button so it broadcasts)...")
        self.motor.connect(device_mac=config.MOTOR_MAC,
                           device_name=config.MOTOR_NAME,
                           device_notification_delay=config.IMU_PERIOD_MS)
        if not self.motor.connected:
            raise RuntimeError("Could not connect to the Double Motor.")
        self.motor.set_notification_callback(self._on_notification)
        self._running = True
        threading.Thread(target=self._command_worker, daemon=True).start()
        print("Paddle connected!")
        self.light("blue")

    def disconnect(self):
        self._running = False
        self._cmds.put(None)
        try:
            self.motor.motor_stop(motor=le.MOTOR_BOTH)
            self.motor.disconnect()
        except Exception:
            pass

    # --------------------------------------------------------------- IMU ---
    def _on_notification(self, _data):
        # The library already parsed the packet into imu_device.
        # (No motor commands allowed in here - only store data.)
        imu = self.motor.imu_device
        sample = (time.time(), imu.accelerometerX, imu.accelerometerY,
                  imu.accelerometerZ, imu.gyroscopeX, imu.gyroscopeY,
                  imu.gyroscopeZ)
        with self._lock:
            self.samples.append(sample)

    def get_samples(self, since=0.0):
        """All IMU samples newer than time `since`."""
        with self._lock:
            return [s for s in self.samples if s[0] > since]

    # ----------------------------------------------------------- feedback --
    def _command_worker(self):
        while self._running:
            job = self._cmds.get()
            if job is None:
                break
            try:
                job()
            except Exception as e:  # never let a BLE hiccup kill the game
                print("paddle command failed:", e)

    def _send(self, job):
        # Drop old feedback if Bluetooth is backed up, so it stays in sync.
        if self._cmds.qsize() < 4:
            self._cmds.put(job)

    def _pulse(self, motor, speed, ms):
        self.motor.motor_run_for_time(ms, motor=motor, speed=speed)

    def haptic(self, kind):
        """kind: 'sweet', 'edge', 'miss', 'warn', 'smash'."""
        M = le
        if kind == "sweet":      # one crisp, strong thump from both motors
            self._send(lambda: self._pulse(M.MOTOR_BOTH, 100, 90))
        elif kind == "smash":    # double thump
            def job():
                self._pulse(M.MOTOR_BOTH, 100, 80)
                time.sleep(0.05)
                self._pulse(M.MOTOR_BOTH, 100, 80)
            self._send(job)
        elif kind == "edge":     # weak, lopsided buzz from ONE motor = bad contact
            self._send(lambda: self._pulse(M.MOTOR_LEFT, 40, 60))
        elif kind == "miss":     # three sad rattles, alternating sides
            def job():
                for m in (M.MOTOR_LEFT, M.MOTOR_RIGHT, M.MOTOR_LEFT):
                    self._pulse(m, 60, 50)
                    time.sleep(0.04)
            self._send(job)
        elif kind == "warn":     # tiny tick: "you're outside your zone"
            self._send(lambda: self._pulse(M.MOTOR_RIGHT, 30, 30))

    def beep(self, frequency=440, pattern=0):
        self._send(lambda: self.motor.beep(pattern=pattern, frequency=int(frequency)))

    def light(self, color, pattern=0):
        colors = {"red": le.LEGO_COLOR_RED, "green": le.LEGO_COLOR_GREEN,
                  "blue": le.LEGO_COLOR_BLUE, "yellow": le.LEGO_COLOR_YELLOW,
                  "purple": le.LEGO_COLOR_PURPLE, "orange": le.LEGO_COLOR_ORANGE}
        c = colors.get(color, le.LEGO_COLOR_BLUE)
        self._send(lambda: self.motor.light_color(c, pattern=pattern))


class FakePaddle:
    """Stand-in paddle. fake_swing() injects a forehand-shaped gyro spike."""

    def __init__(self):
        self.samples = deque(maxlen=500)
        self._lock = threading.Lock()

    def connect(self):
        print("Using FAKE paddle: press SPACE to swing.")

    def disconnect(self):
        pass

    def fake_swing(self):
        t0 = time.time()
        peak_index = config.WINDOW_BEFORE
        sample_count = config.WINDOW_BEFORE + 1 + config.WINDOW_AFTER
        period = config.IMU_PERIOD_MS / 1000
        peak_gyro = max(config.SWING_GYRO_THRESHOLD + 1,
                        min(3000, config.SMASH_GYRO_THRESHOLD))
        with self._lock:
            for i in range(sample_count):
                g = peak_gyro * max(0.0, 1 - abs(i - peak_index) / 5)
                self.samples.append((t0 + (i - peak_index) * period,
                                     0, 0, 1000, 0, 0, g))

    def get_samples(self, since=0.0):
        with self._lock:
            return [s for s in self.samples if s[0] > since]

    def haptic(self, kind):
        print(f"  [haptic: {kind}]")

    def beep(self, frequency=440, pattern=0):
        pass

    def light(self, color, pattern=0):
        pass
