# UNO Q Linux side of the finger lock (computer version).
# Exposes a small web API on port 7000 so the laptop can read the button
# and control the LED and DC motor latch. Requires the Web UI brick in App Lab.

import threading

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

ui = WebUI()
_lock = threading.Lock()
_state = {"pressed": False}


def button_pressed():
    """Called by the sketch whenever the button is pressed."""
    with _lock:
        _state["pressed"] = True
    print("Button pressed", flush=True)
    return True


Bridge.provide("button_pressed", button_pressed)


def api_button():
    """Laptop polls this. Returns whether the button was pressed since last check."""
    with _lock:
        pressed = _state["pressed"]
        _state["pressed"] = False
    return {"pressed": pressed}


def api_leds_on():
    Bridge.call("set_leds", 1)
    return {"ok": True}


def api_leds_off():
    Bridge.call("set_leds", 0)
    return {"ok": True}


def api_unlock():
    Bridge.call("set_lock", 1)
    print("UNLOCK", flush=True)
    return {"ok": True}


def api_lock():
    Bridge.call("set_lock", 0)
    print("LOCK", flush=True)
    return {"ok": True}


def api_fail():
    Bridge.call("fail", 3)
    print("Wrong code", flush=True)
    return {"ok": True}


ui.expose_api("GET", "/api/button", api_button)
ui.expose_api("GET", "/api/leds_on", api_leds_on)
ui.expose_api("GET", "/api/leds_off", api_leds_off)
ui.expose_api("GET", "/api/unlock", api_unlock)
ui.expose_api("GET", "/api/lock", api_lock)
ui.expose_api("GET", "/api/fail", api_fail)

print("Lock app ready", flush=True)
App.run()
