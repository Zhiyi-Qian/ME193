#include <Wire.h>
#include <Arduino_RouterBridge.h>

const int IMU_ADDR = 0x68;

// Wider ranges so running doesn't max out the sensor:
// accel +-8 g  -> 4096 counts per g
// gyro  +-1000 deg/s -> 32.8 counts per deg/s
const float ACC_SCALE = 4096.0;
const float GYRO_SCALE = 32.8;

float gxOff = 0, gyOff = 0, gzOff = 0;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void readRaw(int16_t raw[7]) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(IMU_ADDR, 14);
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
}

void setup() {
  Bridge.begin();
  Wire.begin();
  writeReg(0x6B, 0x00);  // wake up
  writeReg(0x1B, 0x10);  // gyro range +-1000 deg/s
  writeReg(0x1C, 0x10);  // accel range +-8 g
  delay(100);

  // Gyro calibration: keep the sensor STILL for ~1 second at startup
  const int N = 200;
  int16_t raw[7];
  for (int i = 0; i < N; i++) {
    readRaw(raw);
    gxOff += raw[4] / GYRO_SCALE;
    gyOff += raw[5] / GYRO_SCALE;
    gzOff += raw[6] / GYRO_SCALE;
    delay(5);
  }
  gxOff /= N; gyOff /= N; gzOff /= N;
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last < 20) return;  // 50 samples per second
  last = millis();

  int16_t raw[7];
  readRaw(raw);
  float ax = raw[0] / ACC_SCALE, ay = raw[1] / ACC_SCALE, az = raw[2] / ACC_SCALE;
  float gx = raw[4] / GYRO_SCALE - gxOff;
  float gy = raw[5] / GYRO_SCALE - gyOff;
  float gz = raw[6] / GYRO_SCALE - gzOff;

  Bridge.notify("imu", ax, ay, az, gx, gy, gz);
}
