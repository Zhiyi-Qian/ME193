"""
Game logic for LEGO Ping-Pong. No camera, Bluetooth or sound in here, so it
can be unit tested (see tests/test_game.py). Time is always passed in.

States:
  WAIT_START -> (hold AprilTag START) -> READY -> (clap) -> INCOMING
  INCOMING   -> ball arrives -> judged HIT -> OUTGOING, or MISS -> READY
  OUTGOING   -> opponent returns it -> INCOMING, or opponent misses -> READY
"""

import json
import math
import os
import random

import config

WAIT_START, READY, INCOMING, OUTGOING = "WAIT_START", "READY", "INCOMING", "OUTGOING"
OPP_Y = 0.15                 # opponent's end of the table (screen fraction)
AIMS = (0.2, 0.5, 0.8)       # where in your zone the opponent can aim


# ----------------------------------------------------------- Q-learning ----
class OpponentBrain:
    """
    Q-learning opponent. State = which third of your zone your hand is in
    (0 left, 1 middle, 2 right). Action = aim left / middle / right.
    Reward = +1 when you miss, -0.2 when you return it.
    Over time it learns where to send the ball so you miss. The table is
    saved, so it keeps learning across games.
    """

    def __init__(self, path=config.Q_TABLE_PATH, alpha=0.3, gamma=0.5):
        self.path, self.alpha, self.gamma = path, alpha, gamma
        self.q = [[0.0] * 3 for _ in range(3)]
        if path and os.path.exists(path):
            with open(path) as f:
                self.q = json.load(f)

    def choose(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(3)               # explore
        row = self.q[state]
        best = max(row)
        return random.choice([a for a in range(3) if row[a] == best])  # exploit

    def learn(self, s, a, reward, s_next):
        target = reward + self.gamma * max(self.q[s_next])
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    def save(self):
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.q, f)


# ----------------------------------------------------------------- Game ----
class Game:
    def __init__(self, brain=None, aspect=9 / 16):
        self.brain = brain or OpponentBrain()
        self.aspect = aspect          # frame height / width (for distances)
        self.state = WAIT_START
        self.level = 1
        self.streak = 0
        self.record = 0
        self.zone = (0.1, 0.9)        # x limits of the player zone
        self.zone_points = ((0.1, 0.72), (0.9, 0.72))
        self.zone_tag_points = {config.TAG_LEFT_ZONE: (0.1, 0.72),
                                config.TAG_RIGHT_ZONE: (0.9, 0.72)}
        self.zone_from_tags = False
        self.message = "Show the START tag to begin"
        self.swings = []              # (t, label) from the IMU
        self.wrist = None
        self.wrist_hist = []          # (t, wrist) while the ball is close
        self.start_seen_since = None
        self.last_warn = 0.0
        self.opp_x = 0.5
        self.ball = None              # dict describing the current flight
        self.pending = None           # (state, action) waiting for reward
        self.last_result = ""

    # -------------------------------------------------------- inputs -------
    def add_swing(self, t, label):
        self.swings.append((t, label))
        self.swings = self.swings[-20:]

    def set_level(self, level):
        if level != self.level and self.state in (WAIT_START, READY):
            self.level = level
            return [("level", level)]
        return []

    def update_zone(self, tags):
        left = tags.get(config.TAG_LEFT_ZONE)
        right = tags.get(config.TAG_RIGHT_ZONE)
        if left and right:
            self.zone_tag_points = {
                config.TAG_LEFT_ZONE: left["center"],
                config.TAG_RIGHT_ZONE: right["center"],
            }
            self.zone_points = tuple(sorted(
                self.zone_tag_points.values(), key=lambda point: point[0]))
            self.zone = (self.zone_points[0][0], self.zone_points[1][0])
            self.zone_from_tags = True

    def in_zone(self, wrist):
        return wrist is not None and self.zone[0] <= wrist[0] <= self.zone[1]

    def zone_third(self, wrist):
        if wrist is None:
            return 1
        lo, hi = self.zone
        f = (wrist[0] - lo) / max(hi - lo, 1e-6)
        return min(2, max(0, int(f * 3)))

    def serve(self, now):
        if self.state != READY:
            return []
        self.opp_x = 0.5
        self._launch_incoming(now, from_x=0.5)
        return [("serve", None)]

    # -------------------------------------------------------- update -------
    def update(self, now, wrist, tags):
        """Call every frame. Returns a list of (event, value) tuples."""
        self.wrist = wrist
        self.update_zone(tags)
        events = []

        if self.state == WAIT_START:
            if config.TAG_START in tags:
                if self.start_seen_since is None:
                    self.start_seen_since = now
                elif now - self.start_seen_since >= config.TAG_HOLD_S:
                    events += self.start()
            else:
                self.start_seen_since = None

        elif self.state == INCOMING:
            b = self.ball
            if now >= b["t_arrive"] - config.SWING_WINDOW_S:
                self.wrist_hist.append((now, wrist))
            # gentle warning tick if you're out of your zone while it's coming
            if not self.in_zone(wrist) and now - self.last_warn > 0.8:
                self.last_warn = now
                events.append(("warn", None))
            if now >= b["t_arrive"] + config.SWING_WINDOW_S:
                events += self._judge(now)

        elif self.state == OUTGOING:
            b = self.ball
            # Opponent paddle: proportional controller toward where the
            # ball will land, with a speed limit.
            dt = now - b.get("t_last", now)
            b["t_last"] = now
            gain = config.LEVELS[self.level]["opp_gain"]
            v = gain * (b["to"][0] - self.opp_x)
            v = max(-config.OPP_MAX_SPEED, min(config.OPP_MAX_SPEED, v))
            self.opp_x += v * dt
            if now >= b["t_arrive"]:
                if abs(self.opp_x - b["to"][0]) <= config.OPP_REACH:
                    self._launch_incoming(now, from_x=b["to"][0])
                    events.append(("opp_return", None))
                else:
                    self.state = READY
                    self.ball = None
                    self.message = "Opponent missed! Clap to serve again"
                    events.append(("opp_miss", None))
        return events

    def start(self):
        self.state = READY
        self.streak = 0
        self.message = f"Level {self.level}. Clap (or press C) to serve!"
        return [("start", self.level)]

    # -------------------------------------------------------- ball ---------
    def _launch_incoming(self, now, from_x):
        s = self.zone_third(self.wrist)
        eps = config.LEVELS[self.level]["epsilon"]
        a = self.brain.choose(s, eps)
        self.pending = (s, a)
        lo, hi = self.zone
        tx = lo + AIMS[a] * (hi - lo) + random.uniform(-0.04, 0.04)
        travel = config.LEVELS[self.level]["travel_time"]
        self.ball = dict(frm=(from_x, OPP_Y), to=(tx, config.BALL_ARRIVAL_Y),
                         t0=now, t_arrive=now + travel)
        self.wrist_hist = []
        self.state = INCOMING
        self.message = ""

    def ball_screen(self, now):
        """(x, y, radius) of the ball for drawing, or None."""
        b = self.ball
        if b is None:
            return None
        p = (now - b["t0"]) / max(b["t_arrive"] - b["t0"], 1e-6)
        p = min(max(p, 0.0), 1.15)
        x = b["frm"][0] + (b["to"][0] - b["frm"][0]) * p
        y = b["frm"][1] + (b["to"][1] - b["frm"][1]) * p
        y -= 0.12 * math.sin(math.pi * min(p, 1.0))     # little arc
        near = p if self.state == INCOMING else 1 - p     # bigger when close
        return x, y, 0.012 + 0.035 * near

    # -------------------------------------------------------- policy -------
    def _judge(self, now):
        """THE HIT POLICY: decide hit or miss for the ball that just arrived."""
        b = self.ball
        tx, ty = b["to"]
        t_arr = b["t_arrive"]

        # Closest the hand got to the ball during the hit window
        best = None
        for _, w in self.wrist_hist:
            if w is None:
                continue
            d = math.hypot(w[0] - tx, (w[1] - ty) * self.aspect)
            if best is None or d < best[0]:
                best = (d, w)

        swing = None
        for t, label in self.swings:
            if abs(t - t_arr) <= config.SWING_WINDOW_S:
                if swing is None or abs(t - t_arr) < abs(swing[0] - t_arr):
                    swing = (t, label)

        if best is None:
            return self._miss("I can't see your hand", now)
        dist, w = best
        if not self.in_zone(w):
            return self._miss("Out of your zone!", now)
        if swing is None:
            return self._miss("No swing!" if dist <= config.HIT_RADIUS else "Too far & no swing", now)
        if dist > config.HIT_RADIUS:
            return self._miss("Swung, but too far from the ball", now)

        self.swings.remove(swing)       # each swing can only hit one ball
        label = swing[1]
        quality = "sweet" if dist <= config.SWEET_RADIUS else "edge"
        forehand_side = tx > (self.zone[0] + self.zone[1]) / 2
        if config.HAND == "left":
            forehand_side = not forehand_side
        if (label == "forehand" and not forehand_side) or (label == "backhand" and forehand_side):
            quality = "edge"            # wrong stroke for that side = bad contact
        return self._hit(quality, label, now)

    def _hit(self, quality, label, now):
        self.streak += 1
        events = [("hit", (quality, label, self.streak))]
        if self.streak > self.record:
            self.record = self.streak
            events.append(("record", float(self.record)))
        self._reward(-0.2)
        # send it back to the opponent
        travel = config.LEVELS[self.level]["travel_time"] * (0.55 if label == "smash" else 0.8)
        land_x = random.uniform(0.2, 0.8)
        self.ball = dict(frm=self.ball["to"], to=(land_x, OPP_Y), t0=now,
                         t_arrive=now + travel, t_last=now)
        self.state = OUTGOING
        self.last_result = "SMASH!" if label == "smash" else ("Sweet!" if quality == "sweet" else "Edge...")
        return events

    def _miss(self, reason, now):
        self.streak = 0
        self._reward(1.0)
        self.state = READY
        self.ball = None
        self.last_result = reason
        self.message = "Clap (or press C) to serve"
        return [("miss", reason)]

    def _reward(self, r):
        if self.pending:
            s, a = self.pending
            self.brain.learn(s, a, r, self.zone_third(self.wrist))
            self.pending = None
