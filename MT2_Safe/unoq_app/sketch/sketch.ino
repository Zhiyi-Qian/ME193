// UNO Q microcontroller side of the finger lock.
// Hardware: pushbutton on pin 2, one LED on pin 4, DC motor latch through a
// Cytron Maker Drive (M1A on pin 9, M1B on pin 10).
// Python (on the UNO Q's Linux side) tells it what to do through the Bridge,
// and it reports button presses back.

#include <Arduino_RouterBridge.h>

const int BUTTON_PIN = 2;   // pushbutton to GND (uses internal pull-up)
const int LED_PIN    = 4;   // status LED (through a 220 ohm resistor to GND)
const int MOTOR_IN1  = 9;   // Maker Drive header M1A
const int MOTOR_IN2  = 10;  // Maker Drive header M1B

// Motor timing and speed. Tune these for your mechanism:
// too short / too slow = latch only moves part way, too long = motor strains against the end.
const unsigned long MOTOR_MS = 3000;   // run time in milliseconds
const int MOTOR_SPEED = 120;           // PWM duty: 0 = stopped, 255 = full speed

bool isOpen = false;        // assumes the latch is CLOSED when the board starts

// Requests from Python are stored here and carried out in loop().
volatile int ledRequest  = -1;  // 1 on, 0 off
volatile int lockRequest = -1;  // 1 open, 0 locked
volatile int failRequest = 0;   // number of LED flashes for a wrong code

bool set_leds(int on)   { ledRequest = on;    return true; }
bool set_lock(int open) { lockRequest = open; return true; }
bool fail(int flashes)  { failRequest = flashes; return true; }

void motorStop()    { analogWrite(MOTOR_IN1, 0);           analogWrite(MOTOR_IN2, 0); }
void motorForward() { analogWrite(MOTOR_IN1, MOTOR_SPEED); analogWrite(MOTOR_IN2, 0); }
void motorReverse() { analogWrite(MOTOR_IN1, 0);           analogWrite(MOTOR_IN2, MOTOR_SPEED); }

void moveLatch(bool open) {
  if (open == isOpen) return;      // already there: don't run the motor
  // On this build the M1B direction opens the latch. If your latch moves the wrong way,
  // swap these two calls (or swap the motor wires on the M1A/M1B screw terminals).
  if (open) motorReverse(); else motorForward();
  delay(MOTOR_MS);
  motorStop();
  isOpen = open;
}

int lastButton = HIGH;
unsigned long lastChange = 0;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  motorStop();
  digitalWrite(LED_PIN, LOW);

  Bridge.begin();
  Bridge.provide("set_leds", set_leds);
  Bridge.provide("set_lock", set_lock);
  Bridge.provide("fail", fail);
}

void loop() {
  if (ledRequest >= 0) {
    digitalWrite(LED_PIN, ledRequest ? HIGH : LOW);
    ledRequest = -1;
  }

  if (lockRequest >= 0) {
    moveLatch(lockRequest == 1);
    lockRequest = -1;
  }

  // Wrong code: flash the LED quickly, then leave it off.
  if (failRequest > 0) {
    int n = failRequest;
    failRequest = 0;
    for (int i = 0; i < n; i++) {
      digitalWrite(LED_PIN, LOW);  delay(150);
      digitalWrite(LED_PIN, HIGH); delay(150);
    }
    digitalWrite(LED_PIN, LOW);
  }

  // Debounced button: report each new press to Python.
  int b = digitalRead(BUTTON_PIN);
  if (b != lastButton && millis() - lastChange > 50) {
    lastChange = millis();
    lastButton = b;
    if (b == LOW) {
      Bridge.call("button_pressed");
    }
  }
  delay(5);
}
