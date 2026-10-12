"""
Clap-to-serve with pyAudio.

Listens to the microphone in the background. First it measures how loud the
room is for one second, then a "clap" is any short sound much louder than
that (CLAP_FACTOR times the room level, and at least CLAP_MIN_RMS).
"""

import threading
import time

import numpy as np

import config

RATE, CHUNK = 16000, 512


class ClapDetector:
    def __init__(self):
        self.ok = False
        self._event = threading.Event()
        self.level = 0.0
        self._noise = None
        self._samples = []
        self._last = 0.0
        try:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                                       input=True, frames_per_buffer=CHUNK,
                                       stream_callback=self._callback)
            self.ok = True
            print("Microphone on: calibrating room noise for 1 s, stay quiet...")
        except Exception as e:
            print(f"No microphone ({e}). Press C to serve instead.")

    def _callback(self, data, frames, time_info, status):
        import pyaudio
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        rms = float(np.sqrt(np.mean(x * x)))
        self.level = rms
        if self._noise is None:                      # still calibrating
            self._samples.append(rms)
            if len(self._samples) >= RATE // CHUNK:  # about 1 second
                self._noise = float(np.median(self._samples))
                print(f"Room noise level: {self._noise:.4f}")
        else:
            threshold = max(self._noise * config.CLAP_FACTOR, config.CLAP_MIN_RMS)
            now = time.time()
            if rms > threshold and now - self._last > 0.8:
                self._last = now
                self._event.set()
        return (None, pyaudio.paContinue)

    def clapped(self):
        """True once per clap."""
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def close(self):
        if self.ok:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
