"""
car_controller.py

Thin wrapper around the LEGO Education `legoeducation` Python package's
DoubleMotor class. Translates high-level gesture commands (FORWARD,
TURN_LEFT, ...) into `movement_move_tank(speed_left, speed_right)` calls,
and keeps track of a variable "current speed" that the SPEED_UP / SLOW_DOWN
gestures adjust.

Set `simulate=True` (or pass --simulate on the command line, see main.py)
to run without any real hardware connected -- commands are just printed,
which is useful for testing the computer-vision side on its own.
"""

import time

try:
    import legoeducation as le
except ImportError:  # pragma: no cover - only hit if the package isn't installed
    le = None

from gesture_control import FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, SPEED_UP, SLOW_DOWN, HOLD

MIN_SPEED = 20
MAX_SPEED = 100
DEFAULT_SPEED = 40
SPEED_STEP = 2            # how much SPEED_UP/SLOW_DOWN changes speed per update
SPEED_ADJUST_INTERVAL = 0.15  # seconds between speed nudges, so it doesn't rocket to 100 instantly

# How much slower the inside wheel spins during a turn, as a fraction of
# the main speed. 0.0 = pivot in place, 1.0 = no turn at all.
TURN_INSIDE_WHEEL_RATIO = 0.15

# Don't spam the BLE link: only re-send a command if it changed, or if this
# many seconds have passed since the last send (keeps speed updates flowing).
MIN_RESEND_INTERVAL = 0.1


class CarController:
    def __init__(self, simulate=False):
        self.simulate = simulate or le is None
        if simulate and le is None:
            print("[car] 'legoeducation' package not found — running in simulate mode anyway.")
        self.motor = None
        self.speed = DEFAULT_SPEED
        self._last_command_sent = None
        self._last_send_time = 0.0
        self._last_speed_adjust_time = 0.0

    def connect(self):
        if self.simulate:
            print("[car] SIMULATE mode: not connecting to real hardware.")
            return
        self.motor = le.DoubleMotor()
        print("[car] Scanning for the Double Motor hub... power it on and keep it close.")
        self.motor.connect()
        if not self.motor.connected:
            raise RuntimeError(
                "Could not connect to the Double Motor. Make sure it's charged, "
                "powered on, and broadcasting, then try again."
            )
        print("[car] Connected to Double Motor.")
        # Brake to a stop when we're not actively driving, instead of coasting.
        self.motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)

    def disconnect(self):
        if self.simulate or self.motor is None:
            return
        try:
            self.motor.movement_stop()
            self.motor.disconnect()
        except Exception as exc:  # best-effort cleanup
            print(f"[car] Error while disconnecting: {exc}")

    def _adjust_speed(self, command):
        now = time.time()
        if now - self._last_speed_adjust_time < SPEED_ADJUST_INTERVAL:
            return
        self._last_speed_adjust_time = now
        if command == SPEED_UP:
            self.speed = min(MAX_SPEED, self.speed + SPEED_STEP)
        elif command == SLOW_DOWN:
            self.speed = max(MIN_SPEED, self.speed - SPEED_STEP)

    def _wheel_speeds_for(self, command):
        """Return (left, right) wheel speed percentages for a driving command.
        Returns None for commands that aren't a driving command on their own
        (e.g. HOLD, SPEED_UP, SLOW_DOWN keep whatever direction is already
        running)."""
        s = self.speed
        if command == FORWARD:
            return s, s
        if command == BACKWARD:
            return -s, -s
        if command == TURN_LEFT:
            return s * TURN_INSIDE_WHEEL_RATIO, s
        if command == TURN_RIGHT:
            return s, s * TURN_INSIDE_WHEEL_RATIO
        if command == STOP:
            return 0, 0
        return None

    def apply(self, command):
        """Apply a gesture command. Call this once per video frame."""
        if command in (SPEED_UP, SLOW_DOWN):
            self._adjust_speed(command)
            # Re-derive wheel speeds from whatever direction was last active,
            # so the speed change takes effect immediately.
            wheels = self._wheel_speeds_for(self._last_command_sent or STOP)
            effective_command = self._last_command_sent or STOP
        elif command == HOLD:
            return  # keep doing exactly what we were doing, don't resend
        else:
            wheels = self._wheel_speeds_for(command)
            effective_command = command

        if wheels is None:
            return

        now = time.time()
        should_resend = (
            effective_command != self._last_command_sent
            or command in (SPEED_UP, SLOW_DOWN)
            or (now - self._last_send_time) > MIN_RESEND_INTERVAL
        )
        if not should_resend:
            return

        left, right = wheels
        self._send(effective_command, left, right)
        self._last_command_sent = effective_command
        self._last_send_time = now

    def _send(self, command, left_speed, right_speed):
        left_speed, right_speed = round(left_speed), round(right_speed)
        if self.simulate:
            print(f"[car][sim] {command:<10} speed={self.speed:>3}  L={left_speed:>4} R={right_speed:>4}")
            return
        if left_speed == 0 and right_speed == 0:
            self.motor.movement_stop(blocking=False)
        else:
            self.motor.movement_move_tank(left_speed, right_speed, blocking=False)
