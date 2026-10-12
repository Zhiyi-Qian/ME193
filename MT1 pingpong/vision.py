"""
Camera processing: MediaPipe pose (where is the paddle hand?) and AprilTags
(play-zone edges, level, start).

Detection runs on the raw camera frame (AprilTags can't be read in a mirror
image, and MediaPipe's left/right labels assume a normal image). Coordinates are
then converted to the MIRRORED image the player sees, so moving right moves
right on screen. All coordinates are normalized 0..1.
"""

import os
import shutil
import ssl
import tempfile
import urllib.request

import certifi
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import config

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# MediaPipe pose landmark numbers
WRIST = {"left": 15, "right": 16}
ELBOW = {"left": 13, "right": 14}
SHOULDER = {"left": 11, "right": 12}


def ensure_model(path=config.POSE_MODEL_PATH):
    if not os.path.exists(path):
        print("Downloading the MediaPipe pose model (one time, ~6 MB)...")
        with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(path)), delete=False) as temp:
            temp_path = temp.name
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(MODEL_URL, context=context) as response:
                with open(temp_path, "wb") as model_file:
                    shutil.copyfileobj(response, model_file)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return path


class Tracker:
    def __init__(self):
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1)
        self.pose = vision.PoseLandmarker.create_from_options(opts)
        self.tags = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            cv2.aruco.DetectorParameters())
        self.last_ts = -1

    def process(self, frame_bgr, t_ms):
        """
        frame_bgr: the RAW camera frame (not mirrored - AprilTags can't be read
                   in a mirror image!).
        Returns (mirrored_frame, info) where info has:
            wrist:  (x, y) or None      elbow, shoulder: same
            tags:   {id: dict(center=(x, y), size=pixels)}
        """
        h, w = frame_bgr.shape[:2]
        info = {"wrist": None, "elbow": None, "shoulder": None, "tags": {}}

        # --- AprilTags on the raw frame, then mirror the x coordinate
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.tags.detectMarkers(gray)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                pts = c.reshape(4, 2)
                cx, cy = pts.mean(0)
                size = cv2.norm(pts[0] - pts[1])
                info["tags"][int(i)] = dict(center=(float(1 - cx / w), float(cy / h)), size=float(size))

        # --- Pose on the RAW frame too (MediaPipe's left/right labels are only
        #     correct on a non-mirrored image), then mirror the x coordinate.
        t_ms = max(int(t_ms), self.last_ts + 1)   # timestamps must increase
        self.last_ts = t_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            for key, table in (("wrist", WRIST), ("elbow", ELBOW), ("shoulder", SHOULDER)):
                p = lm[table[config.HAND]]
                if p.visibility > 0.5:
                    info[key] = (1 - p.x, p.y)

        mirrored = cv2.flip(frame_bgr, 1)   # what the player sees
        return mirrored, info

    def close(self):
        self.pose.close()


def level_from_tag_size(size_px):
    big, small = config.LEVEL_SIZE_THRESHOLDS
    if size_px > big:
        return 1
    if size_px > small:
        return 2
    return 3
