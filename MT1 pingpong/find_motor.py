"""
No connection card? Run this to list the Double Motors nearby.
Press the button on YOUR motor first. Copy its address into MOTOR_MAC in
config.py so the game never grabs a classmate's motor.
"""

import legoeducation as le

motor = le.DoubleMotor()
print("Scanning for 5 seconds...")
found = motor.search(timeout=5)
if not found:
    print("Nothing found. Is the motor on and blinking (broadcasting)?")
else:
    for i, d in enumerate(found):
        print(f"{i}: {d}")
    print("\nTip: turn your motor off and scan again - the one that disappears is yours.")
