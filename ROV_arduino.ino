#include <Servo.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <PID_v1.h>



// --------- Động cơ ---------
Servo motors[6];
const int motorPins[6] = {9, 8, 10, 11, 12, 13};
const int stopSpeeds[6] = {1500, 1480, 1480, 1000, 1500, 1500};
int currentSpeeds[6];
int targetSpeeds[6];

const int SPEED_STEP = 10;
const int MIN_SPEED_STEP = 5;
const int MAX_SPEED_STEP = 20;
const unsigned long MOTOR_INTERVAL = 20;
unsigned long prevMotorMillis = 0;

// --------- IMU ---------
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
imu::Vector<3> imu_offset;
bool bno_ok = true;

const float OffSet = 0.483;  // Điện áp tương ứng với 0 KPa
float V, P, depth;
float initialPressure = -1;  // Áp suất ban đầu, dùng làm mốc 0m
unsigned long previousMillis = 0;
const unsigned long interval = 500;  // thời gian chờ giữa 2 lần đo (ms)

// --------- PID Roll ---------
double rollInput = 0, rollOutput = 0, rollSetpoint = 0;
double roll_Kp = 0.0, roll_Ki = 0.0, roll_Kd = 0.0;
PID rollPID(&rollInput, &rollOutput, &rollSetpoint, roll_Kp, roll_Ki, roll_Kd, DIRECT);

// --------- PID Yaw ---------
double yawInput = 0, yawOutput = 0, yawSetpoint = 0;
double yaw_Kp = 0.0, yaw_Ki = 0.0, yaw_Kd = 0.0;
PID yawPID(&yawInput, &yawOutput, &yawSetpoint, yaw_Kp, yaw_Ki, yaw_Kd, DIRECT);

// --------- PID Độ sâu (Dive PID) ---------
double depthInput = 0, depthOutput = 0, depthSetpoint = 0;
double depth_Kp = 0.0, depth_Ki = 0.0, depth_Kd = 0.0;
PID depthPID(&depthInput, &depthOutput, &depthSetpoint, depth_Kp, depth_Ki, depth_Kd, DIRECT);

bool yaw_pid_mode = false;

int dive_base_pwm_left = 1500;
int dive_base_pwm_right = 1500;

bool forward_mode = false;
bool pid_ready = false; // Biến này cho PID Roll
bool depth_pid_ready = false; // Biến mới cho PID độ sâu
int forward_base_pwm = 1500;

// --------- Calibration (non-blocking) ---------
bool calibrating = false;
int calibSampleCount = 0;
const int calibSamples = 50;
unsigned long calibPrevMillis = 0;
const unsigned long CALIB_INTERVAL = 20;
imu::Vector<3> calibSum(0, 0, 0);


// --------- Chế độ Tự động Xoay (CẢI TIẾN) ---------
bool auto_turn_mode = false;
double auto_turn_target_yaw = 0.0;
const float TURN_ANGLE_THRESHOLD = 10.; // Ngưỡng sai số góc để dừng (độ)
const int TURN_SPEED_PWM = 1600;        // Tốc độ PWM khi xoay, bạn có thể điều chỉnh

// =======================================================
void setup() {
 Serial.begin(115200);
 for (int i = 0; i < 6; i++) {
  motors[i].attach(motorPins[i]);
  currentSpeeds[i] = targetSpeeds[i] = stopSpeeds[i];
  motors[i].writeMicroseconds(currentSpeeds[i]);
 }

 if (!bno.begin()) {
  Serial.println("Khong tim thay BNO055!");
  bno_ok = false;
 } else {
  bno_ok = true;
  bno.setExtCrystalUse(true);
  sensors_event_t event;
  bno.getEvent(&event);
  imu_offset = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  Serial.println("Da tim thay BNO055.");
 }

 rollPID.SetMode(AUTOMATIC);
 rollPID.SetOutputLimits(-300, 300);

 yawPID.SetMode(AUTOMATIC);
 yawPID.SetOutputLimits(-300, 300);
 
 // Thiết lập PID độ sâu
 depthPID.SetMode(AUTOMATIC);
 depthPID.SetOutputLimits(-300, 300); // Giới hạn đầu ra để điều khiển động cơ lặn

 Serial.println("ROV Ready.");
}

// =======================================================
void loop() {
 handleSerialCommands();

 if (calibrating) handleCalibration();

 if (forward_mode && bno_ok && !calibrating) {
  sensors_event_t event;
  bno.getEvent(&event);
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  rollInput = normalizeAngle(euler.x() - imu_offset.x());
  rollPID.Compute();
  
  int pwm_left = constrain(forward_base_pwm + rollOutput -30, 1600, 1800);
  int pwm_right = constrain(forward_base_pwm - rollOutput, 1600, 1800);
  int pwm_right_3 = constrain(1.8 *  pwm_right - 1600,1100,2000);
  setTargetSpeed(4, pwm_left);  // Trái sau
  setTargetSpeed(5, pwm_left);  // Trái trước
  setTargetSpeed(0, pwm_right); // Phải trước
  setTargetSpeed(3, pwm_right_3); // Phải sau
 }

 if (yaw_pid_mode && bno_ok && !calibrating) {
  int pwm_left, pwm_right;
  sensors_event_t event;
  bno.getEvent(&event);
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  yawInput = normalizeAngle(euler.z() - imu_offset.z());
  yawPID.Compute();
  if(dive_base_pwm_left > 1400 && dive_base_pwm_right>1400)
  { 
    pwm_left = constrain(dive_base_pwm_left + yawOutput, 1400, 1800);
    pwm_right = constrain(dive_base_pwm_right - yawOutput, 1400, 1800);
  }
  else
  {
    pwm_left = constrain(dive_base_pwm_left + yawOutput, 1100, 1400);
    pwm_right = constrain(dive_base_pwm_right - yawOutput, 1100, 1400);
  }
  
  setTargetSpeed(1, pwm_left); // Motor lặn 1
  setTargetSpeed(2, pwm_right); // Motor lặn 2
 }
 
 // Xử lý PID độ sâu
 if (depth_pid_ready && !calibrating) {
  depthInput = depth; // Đầu vào PID là độ sâu hiện tại
  depthPID.Compute();
  
  // Điều khiển 2 động cơ lặn (motor 1 và motor 2)
  // Giả sử động cơ 1 là bên trái, động cơ 2 là bên phải
  // Nếu depthOutput dương, ROV cần lặn sâu hơn (tăng lực đẩy xuống)
  // Nếu depthOutput âm, ROV cần nổi lên (giảm lực đẩy xuống hoặc tăng lực đẩy lên)
  int motor1_pwm = constrain(stopSpeeds[1] + depthOutput - 100, 1480, 1800);
  int motor2_pwm = constrain(stopSpeeds[2] - depthOutput, 1100, 1480); // Cả hai động cơ cùng điều khiển độ sâu

  setTargetSpeed(1, motor1_pwm);
  setTargetSpeed(2, motor2_pwm);
 }

  if (auto_turn_mode && bno_ok && !calibrating) {
   handleAutoTurn();
 }


 unsigned long currentMillis = millis();
 if (millis() - prevMotorMillis >= MOTOR_INTERVAL) {
  prevMotorMillis = millis();
  updateMotorSpeeds();
 }

 if (currentMillis - previousMillis >= interval) {
  previousMillis = currentMillis;
  readPressureAndDepth(); // Gọi chương trình con
  handleIMUData();
 }
}

// =======================================================
// -------------------- Xử lý Serial ---------------------
void handleSerialCommands() {
 if (!Serial.available()) return;

 String input = Serial.readStringUntil('\n');
 input.trim();


 if (input.startsWith("dir:")) {
  forward_mode = false;
  depth_pid_ready = false; // Tắt PID độ sâu khi điều khiển thủ công
  Serial.println("Nhận lệnh Dir");
  handleDirCommand(input.substring(4));
 }
 else if (input.startsWith("dive:")) {
  forward_mode = false;
  depth_pid_ready = false; // Tắt PID độ sâu khi điều khiển thủ công
  Serial.println("Nhận lệnh 2 DIVE_MOTOR");
  handleDiveCommand(input.substring(5));
 }
 else if (input == "stop") {
  forward_mode = false;
  yaw_pid_mode = false;
  depth_pid_ready = false; 
  Serial.println("DỪNG TẤT CẢ ĐỘNG CƠ");
  stopAllMotors();
 }
 else if (input == "imu") {
  handleIMUData();
 }
 else if (input == "calib") {
  
  startCalibration();
  Serial.println("ĐÃ CALIB CÁC GÓC IMU");
 }
 else if (input.startsWith("setpid:")) {
  double kp, ki, kd;
  if (sscanf(input.c_str(), "setpid:%lf %lf %lf", &kp, &ki, &kd) == 3) {
   roll_Kp = kp; roll_Ki = ki; roll_Kd = kd;
   rollPID.SetTunings(kp, ki, kd);
   pid_ready = true;
   Serial.println("Roll PID updated.");
  } else Serial.println("Roll PID parse error.");
 }
 else if (input.startsWith("forward:")) {
  if (pid_ready) {
    auto_turn_mode = false;
    double pwm;
    double angle = 0;
    if (sscanf(input.c_str(), "forward:%lf angle:%lf", &pwm, &angle) >= 1) {
      auto_turn_mode = false;
      forward_base_pwm = constrain((int)pwm, 1480, 1800);
      if (bno_ok) {
        sensors_event_t event;
        bno.getEvent(&event);
        imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        rollSetpoint = normalizeAngle(euler.x() - imu_offset.x()+ angle);
        Serial.print("Forward mode: Roll SETPOINT set to ");
        Serial.println(rollSetpoint);
      }
      forward_mode = true;
      Serial.println("FORWARD mode ON");
    } else {
      Serial.println("Invalid forward command format");
    }
  } else Serial.println("Roll PID not set. Ignoring forward.");
 }

 else if (input.startsWith("pidyaw:")) {
  double kp, ki, kd;
  int base1, base2;
  if (sscanf(input.c_str(), "pidyaw:%lf %lf %lf %d", &kp, &ki, &kd, &base1, &base2) == 4) {
   yaw_Kp = kp; yaw_Ki = ki; yaw_Kd = kd;
   yawPID.SetTunings(kp, ki, kd);
   dive_base_pwm_left = constrain(base1, 1000, 1800);
   dive_base_pwm_right = constrain(base2, 1000, 1800);
   yaw_pid_mode = true;
   depth_pid_ready = false; // Tắt PID độ sâu khi ở chế độ Yaw PID
   Serial.println("Yaw PID mode ON");
  } else Serial.println("Yaw PID parse error.");
 }
 else if (input == "pause") { // Lệnh mới để kích hoạt PID độ sâu
  stopAllMotors(); // Dừng tất cả động cơ trước
  forward_mode = false;
  yaw_pid_mode = false;
  // Đặt setpoint cho độ sâu hiện tại
  depthSetpoint = depth; 
  depthPID.SetMode(AUTOMATIC); // Bật chế độ PID tự động cho độ sâu
  depth_pid_ready = true;
  Serial.print("Đứng yên tại độ sâu "); Serial.print(depthSetpoint); Serial.println("m. PID độ sâu ON.");
 }
 else if (input.startsWith("setdepthpid:")) { // Lệnh để cài đặt Kp, Ki, Kd cho PID độ sâu
  double kp, ki, kd;
  
  if (sscanf(input.c_str(), "setdepthpid:%lf %lf %lf", &kp, &ki, &kd) == 3) {
   depth_Kp = kp; depth_Ki = ki; depth_Kd = kd;
   depthPID.SetTunings(kp, ki, kd);
   depth_pid_ready = true;
   depthSetpoint = depth; 
   depthPID.SetMode(AUTOMATIC);
   //Serial.println("Depth PID tunings updated.");
  } else Serial.println("Depth PID tunings parse error.");
 }
  // Thêm chức năng điều khiển rẽ trái
  else if (input == "turn_left") {
    forward_mode = false;
     startAutoTurn(-90.0); // Bắt đầu xoay tự động -90 độ
  }
  // Thêm chức năng điều khiển rẽ phải
  else if (input == "turn_right") {
    forward_mode = false;
    startAutoTurn(90.0); // Bắt đầu xoay tự động +90 độ
  }
 else {
  Serial.println("Unknown command.");
 }
}

// =======================================================
// -------------------- Calibration ----------------------
void startCalibration() {
 if (!bno_ok) {
  Serial.println("CALIB:IMU_NOT_READY");
  return;
 }
 calibrating = true;
 calibSampleCount = 0;
 calibSum = imu::Vector<3>(0, 0, 0);
 calibPrevMillis = millis();
 Serial.println("CALIB:START");
}

void handleCalibration() {
 if (millis() - calibPrevMillis < CALIB_INTERVAL) return;

 calibPrevMillis = millis();
 imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

 if (isnan(euler.x()) || isnan(euler.y()) || isnan(euler.z())) {
  Serial.println("CALIB:FAILED_NAN");
  calibrating = false;
  return;
 }

 calibSum = imu::Vector<3>(
 calibSum.x() + euler.x(),
 calibSum.y() + euler.y(),
 calibSum.z() + euler.z()
 );
 calibSampleCount++;

 if (calibSampleCount >= calibSamples) {
  imu_offset = calibSum * (1.0f / calibSamples);
  Serial.println("CALIB:DONE");
  Serial.print("OFFSET: Roll="); Serial.print(imu_offset.x());
  Serial.print(", Pitch="); Serial.print(imu_offset.y());
  Serial.print(", Yaw="); Serial.println(imu_offset.z());
  calibrating = false;
 }
}

// =======================================================
// ------------------- Motor & IMU -----------------------
void updateMotorSpeeds() {
 for (int i = 0; i < 6; i++) {
  int diff = targetSpeeds[i] - currentSpeeds[i];
  int stop = stopSpeeds[i];

  if ((currentSpeeds[i] - stop) * (targetSpeeds[i] - stop) < 0) {
   if (abs(diff) > SPEED_STEP)
    currentSpeeds[i] += (currentSpeeds[i] > stop) ? -SPEED_STEP : SPEED_STEP;
   else
    currentSpeeds[i] = stop;
  } else {
   int step = constrain(abs(diff) / 4, MIN_SPEED_STEP, MAX_SPEED_STEP);
   currentSpeeds[i] += (diff > 0) ? step : -step;
   if ((diff > 0 && currentSpeeds[i] > targetSpeeds[i]) ||
     (diff < 0 && currentSpeeds[i] < targetSpeeds[i]))
    currentSpeeds[i] = targetSpeeds[i];
  }

  motors[i].writeMicroseconds(currentSpeeds[i]);
 }
}

void stopAllMotors() {
 for (int i = 0; i < 6; i++)
  setTargetSpeed(i, stopSpeeds[i]);
 Serial.println("All motors stopped.");
}

void setTargetSpeed(int motorIndex, int pwm) {
 pwm = constrain(pwm, 1000, 2000);
 targetSpeeds[motorIndex] = pwm;
}

void handleDirCommand(String data) {
 int pwm[4];
 if (sscanf(data.c_str(), "%d %d %d %d", &pwm[0], &pwm[1], &pwm[2], &pwm[3]) == 4) {
  int idxs[4] = {0, 3, 4, 5};
  pwm[1] = constrain(1.8 * pwm[1] -1600,1100,2000);
  for (int i = 0; i < 4; i++) setTargetSpeed(idxs[i], pwm[i]);
  Serial.println("DIR command applied.");
 } else Serial.println("DIR parse failed!");
}

void handleDiveCommand(String data) {
 int pwm[2];
 if (sscanf(data.c_str(), "%d %d", &pwm[0], &pwm[1]) == 2) {
  setTargetSpeed(1, pwm[0]);
  setTargetSpeed(2, pwm[1]);
  Serial.println("DIVE command applied.");
 } else Serial.println("DIVE parse failed!");
}

void handleIMUData() {
 if (!bno_ok) return;
 imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
 if (isnan(euler.x()) || isnan(euler.y()) || isnan(euler.z())) {
  Serial.println("ERROR:IMU_NAN");
  return;
 }

 float roll = normalizeAngle(euler.x() - imu_offset.x());
 float pitch = normalizeAngle(euler.y() - imu_offset.y());
 float yaw = normalizeAngle(euler.z() - imu_offset.z());

 Serial.print("IMU:");
 Serial.print(roll); Serial.print(",");
 Serial.print(pitch); Serial.print(",");
 Serial.println(yaw);
}

float normalizeAngle(float angle) {
 while (angle > 180) angle -= 360;
 while (angle < -180) angle += 360;
 return angle;
}

void startAutoTurn(double angleToTurn) {
  if (!bno_ok) {
    Serial.println("TURN_ERROR: IMU not ready");
    return;
  }
  // Tắt các chế độ khác
  forward_mode = false;

  // Lấy góc hiện tại và tính toán góc mục tiêu
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  double current_yaw = normalizeAngle(euler.x() - imu_offset.x());
  auto_turn_target_yaw = normalizeAngle(current_yaw + angleToTurn); // góc mục tiêu

  auto_turn_mode = true;
  //Serial.print("AUTO_TURN: START. Target Yaw: ");
  //Serial.println(auto_turn_target_yaw);
}

void handleAutoTurn() {
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  double current_yaw = normalizeAngle(euler.x() - imu_offset.x()); 

  // Tính sai số góc ngắn nhất (xử lý trường hợp vượt 180/-180 độ)
  double angle_diff = auto_turn_target_yaw - current_yaw;
  angle_diff = normalizeAngle(angle_diff);

  // Kiểm tra nếu đã đến đích
  if (abs(angle_diff) <= TURN_ANGLE_THRESHOLD) {
    auto_turn_mode = false;
    Serial.println("AUTO_TURN: DONE");
  } else {
    // Nếu chưa đến đích, tiếp tục xoay
    if (angle_diff > 0) { // Cần xoay phải (góc hiện tại < góc mục tiêu)
      
      setTargetSpeed(0, stopSpeeds[0]); 
      setTargetSpeed(3, stopSpeeds[3]); 
      setTargetSpeed(4, TURN_SPEED_PWM); // Động cơ 4, 5 chạy 
      setTargetSpeed(5, TURN_SPEED_PWM); 
    } else { // Cần xoay trái (góc hiện tại > góc mục tiêu)
      
      setTargetSpeed(4, stopSpeeds[4]); 
      setTargetSpeed(5, stopSpeeds[5]); 
      setTargetSpeed(0, TURN_SPEED_PWM); // Động cơ 0, 3 chạy 
      setTargetSpeed(3, TURN_SPEED_PWM); 
    }
  }
}


void readPressureAndDepth() {
 // Bước 1: Đọc điện áp từ cảm biến
 V = analogRead(A0) * 3.3 / 4096.0;

 // Bước 2: Tính áp suất hiện tại từ điện áp đọc được (KPa)
 P = (V - OffSet) * 250;

 // Bước 3: Ghi lại áp suất ban đầu tại thời điểm khởi động (chỉ 1 lần)
 if (initialPressure < 0) {
  initialPressure = P;
 }

 // Bước 4: Tính độ sâu (m) từ chênh lệch áp suất (P - initialPressure)
 float deltaPressure = P - initialPressure;  // KPa
 float depth = deltaPressure / 9.81; // m
 

 // Hiển thị kết quả
 Serial.print("Depth Data: ");
 Serial.print(V, 3);
 Serial.print(",");
 Serial.print(depthSetpoint, 3);
 Serial.print(",");
 Serial.print(depth, 3);
 Serial.println();
}