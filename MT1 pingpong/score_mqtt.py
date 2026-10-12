"""
Posts the record number of continuous hits to MQTT as a float, e.g. "7.0".
Publishes immediately when the record changes and re-sends every
MQTT_HEARTBEAT_S seconds so anyone subscribing late still sees it.
"""

import time

import paho.mqtt.client as mqtt

import config


class ScorePublisher:
    def __init__(self):
        self.ok = False
        self.record = 0.0
        self.last_sent = 0.0
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=30)
            self.client.loop_start()     # network runs on its own thread
            self.ok = True
            print(f"MQTT connected: {config.MQTT_BROKER} -> {config.MQTT_TOPIC}")
        except Exception as e:
            print(f"MQTT not connected ({e}). The game still runs.")

    def publish(self, record, force=False):
        self.record = float(record)
        if self.ok:
            self.client.publish(config.MQTT_TOPIC, f"{self.record:.1f}", qos=1, retain=True)
            self.last_sent = time.time()

    def tick(self):
        if self.ok and time.time() - self.last_sent > config.MQTT_HEARTBEAT_S:
            self.publish(self.record)

    def close(self):
        if self.ok:
            self.client.loop_stop()
            self.client.disconnect()
