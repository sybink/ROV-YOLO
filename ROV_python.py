import serial
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from picamera2 import Picamera2
from libcamera import Transform
import libcamera
from ultralytics import YOLO
from PIL import Image, ImageTk
import time
import cv2
import math
import queue
import numpy as np

# Constants
FOCAL_LENGTH = 1000
OBJECT_REAL_HEIGHT = 21
FOV_DEG = 66
IMAGE_INPUT_WIDTH = 480
FOV_RAD = math.radians(FOV_DEG)
PIXEL_TO_CM = (2 * math.tan(FOV_RAD / 2)) / IMAGE_INPUT_WIDTH

class SerialMain:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"[Serial] Connected to {self.ser.name}")
        except serial.SerialException as e:
            messagebox.showerror("Serial Error", f"Could not open serial port: {e}")
            exit()

        self.roll_data, self.pitch_data, self.yaw_data = [], [], []
        self.imu_time_data = []
        self.voltage_data, self.pressure_data, self.depth_data = [], [], []
        self.pressure_time_data = []
        self.start_time = time.time()

        self.log_queue = queue.Queue()
        self.thread = threading.Thread(target=self.read_serial_data, daemon=True)
        self.thread.start()

    def log(self, msg):
        self.log_queue.put(msg)

    def send(self, command):
        if self.ser and self.ser.is_open:
            self.ser.write((command + '\n').encode())
            self.log(f"Sent: {command}")

    def read_serial_data(self):
        max_data_points = 200
        while True:
            try:
                self.ser.write(b"imu\n")
                while self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("IMU:"):
                        self.parse_imu(line, max_data_points)
                    elif line.startswith("Depth Data:"):
                        self.parse_pressure(line, max_data_points)
                    elif line:
                        self.log(f"[Serial Unknown] {line}")
                time.sleep(0.05)
            except Exception as e:
                self.log(f"[Serial Error] {e}")
                time.sleep(1)

    def parse_imu(self, line, max_points):
        try:
            parts = line[4:].split(",")
            roll, pitch, yaw = map(float, parts[:3])
            elapsed = time.time() - self.start_time
            self.roll_data.append(roll)
            self.pitch_data.append(pitch)
            self.yaw_data.append(yaw)
            self.imu_time_data.append(elapsed)
            if len(self.roll_data) > max_points:
                self.roll_data.pop(0)
                self.pitch_data.pop(0)
                self.yaw_data.pop(0)
                self.imu_time_data.pop(0)
        except Exception as e:
            self.log(f"[IMU Parse Error] {line}")

    def parse_pressure(self, line, max_points):
        try:
            parts = line.replace("Depth Data: ", "").split(",")
            voltage, pressure, depth = map(float, parts[:3])
            elapsed = time.time() - self.start_time
            self.voltage_data.append(voltage)
            self.pressure_data.append(pressure)
            self.depth_data.append(depth)
            self.pressure_time_data.append(elapsed)
            if len(self.voltage_data) > max_points:
                self.voltage_data.pop(0)
                self.pressure_data.pop(0)
                self.depth_data.pop(0)
                self.pressure_time_data.pop(0)
        except Exception as e:
            self.log(f"[Pressure Parse Error] {line}")

class AiMain:
    def __init__(self, serial_main): # Thêm serial_main vào AI class
        self.serial_main = serial_main
        self.yolo_model = YOLO("best480_chatluong_ncnn_model")
        self.picam2 = Picamera2()
        transform = libcamera.Transform(vflip=True, hflip=True)
        self.picam2.preview_configuration.transform = transform
        config = self.picam2.create_preview_configuration(
            main={"size": (IMAGE_INPUT_WIDTH, IMAGE_INPUT_WIDTH), "format": "RGB888"},
            transform=transform
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.model_running = False
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_time = time.monotonic()
        self.fps = 0.0
        self.frame_counter = 0
        self.fps_update_interval = 1.0  # seconds
        self.time_monotonic_check = time.monotonic()
        self.fps_timer = time.monotonic()
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        self.state = "IDLE"  # Trạng thái hoạt động: IDLE, SEARCHING, ACQUIRING_TARGET, TRACKING

        # 2. Biến cho việc khóa mục tiêu (Target Acquisition)
        self.last_known_angle = 0
        self.last_known_distance = 0
        self.acquisition_timer = 0
        self.ACQUISITION_TIMEOUT = 3.0 # Giây

        # 3. Biến cho mẫu tìm kiếm (Search Pattern)
        self.search_phase = "SPIRAL"  # 'SPIRAL' hoặc 'GRID'
        self.search_step_index = 0
        self.search_state_timer = 0
        self.search_target_yaw = None
        self.SPIRAL_PATTERN = [
            ('turn', 90), ('forward', 4), ('turn', -90), ('turn', 90),
            ('forward', 4), ('turn', -90), ('turn', 90), ('forward', 4),
            ('turn', -90), ('turn', -90), ('forward', 12),('turn', 90), ('forward', 4),
        ]*3
        self.GRID_PATTERN = [
            ('forward', 8), ('turn', 90), ('forward', 3), ('turn', 90),
            ('forward', 8), ('turn', -90), ('forward', 3), ('turn', -90),
        ] * 2 # Lặp lại 2 lần cho khu vực lớn hơn


        # 4. Biến cho việc giữ độ sâu
        self.depth_hold_enabled = False
        self.target_depth = None
       



        # Thêm các thuộc tính để lưu giá trị PID và PWM
        self.pwm_base = 1500 # Giá trị mặc định cho PWM cơ bản
        self.kp_roll, self.ki_roll, self.kd_roll = 0.0, 0.0, 0.0
        self.kp_yaw, self.ki_yaw, self.kd_yaw = 0.0, 0.0, 0.0
        self.pwm_dive_base1, self.pwm_dive_base2 = 1500, 1500
        self.kp_depth, self.ki_depth, self.kd_depth = 0.0, 0.0, 0.0
        self.log_text_widget = None # Sẽ được thiết lập bởi GuiMain
        self.ai_entry_forward_pwm = None

    def set_log_widget(self, widget):
        self.log_text_widget = widget

    def set_pwm_entry_widget(self, widget):
        """Nhận và lưu widget ô nhập PWM từ GuiMain."""
        self.ai_entry_forward_pwm = widget

  
    def log(self, msg, tag='info'):
        # Gửi thông điệp log tới luồng chính thông qua queue của SerialMain để đảm bảo thread-safe
        self.serial_main.log(msg)
    

    def send_dir(self, values):
        try:
            command = f"dir:{' '.join(map(str, values))}"
            self.serial_main.send(command)
            self.log(f"[AI Control] Sent DIR: {command}")
        except Exception as e:
            self.log(f"[AI Control Error] Could not send DIR: {e}", 'error')

    def send_dive(self, values):
        try:
            command = f"dive:{' '.join(map(str, values))}"
            self.serial_main.send(command)
            self.log(f"[AI Control] Sent DIVE: {command}")
        except Exception as e:
            self.log(f"[AI Control Error] Could not send DIVE: {e}", 'error')

    def send_stop(self):
        self.serial_main.send("stop")
        self.log("[AI Control] Sent STOP command.")

    def send_calibrate_command(self):
        self.serial_main.send("calib")
        self.log("[AI Control] Sent CALIBRATE command.")

   
    def send_turn_left_command(self):
        #pwm = int(self.ai_entry_forward_pwm.get())
        #self.serial_main.send(f"forward:{pwm} angle:-90.0")
        self.serial_main.send("turn_left")
        self.log("[AI Control] Sent TURN_LEFT command.")

    def send_turn_right_command(self):
        #pwm = int(self.ai_entry_forward_pwm.get())
        #self.serial_main.send(f"forward:{pwm} angle:90.0")
        self.serial_main.send("turn_right")
        self.log("[AI Control] Sent TURN_RIGHT command.")

    def send_pid_values(self):
        try:
            kp_yaw, ki_yaw, kd_yaw = 4.3, 0.2, 0.1
            pwm = int(self.ai_entry_forward_pwm.get())
            self.serial_main.send(f"setpid:{kp_yaw} {ki_yaw} {kd_yaw}")
            self.serial_main.send(f"forward:{pwm} angle:0.0")
            self.log(f"[AI Control] Sent PID Roll: Kp={kp_yaw}, Ki={ki_yaw}, Kd={kd_yaw}, PWM Base={pwm}")
        except ValueError:
            self.log("[AI Control Error] Invalid PID or PWM value for Roll.", 'error')
            
    def send_pid_roll(self,pwm1,pwm2):
        try:
            kp_roll, ki_roll, kd_roll = 4.0, 0.0, 0.0
            self.serial_main.send(f"pidyaw:{kp_roll} {ki_roll} {kd_roll} {pwm1} {pwm2}")
            self.log(f"[AI Control] Sent PID Yaw: Kp={kp_roll}, Ki={ki_roll}, Kd={kd_roll}, PWM1={pwm1}, PWM2={pwm2}")
            self.log_text_widget.insert(tk.END, f"[AI Control] Sent PID Yaw: Kp={kp_roll}, Ki={ki_roll}, Kd={kd_roll}, PWM1={pwm1}, PWM2={pwm2}\n", 'info')
        except ValueError:
            self.log("[AI Control Error] Invalid PID or PWM value for Yaw.", 'error')

    def send_dive_command(self):
        """Gửi lệnh LẶN sử dụng hàm send_pid_roll."""
        # Yêu cầu: pwm1 và pwm2 = 1700
        self.send_pid_roll(pwm1=1700, pwm2=1700)

    def send_surface_command(self):
        """Gửi lệnh NỔI sử dụng hàm send_pid_roll."""
        # Yêu cầu: pwm1 = 1300, pwm2 = 1200
        self.send_pid_roll(pwm1=1200, pwm2=1200)

    def send_depth_pid_values(self):
        try:
            kp_depth, ki_depth, kd_depth = 4.0, 0.1, 0.1
            self.serial_main.send(f"setdepthpid:{kp_depth} {ki_depth} {kd_depth}")
            self.log(f"[AI Control] Sent PID Depth: Kp={kp_depth}, Ki={ki_depth}, Kd={kd_depth}")
        except ValueError:
             self.log("[AI Control Error] Invalid PID value for Depth.", 'error')

    def calculate_distance_and_angle(self, bbox, img_shape):
        x_min, y_min, x_max, y_max = bbox
        h_px = y_max - y_min
        d = float('inf')

        min_h_px = IMAGE_INPUT_WIDTH * 0.85
        max_h_px = IMAGE_INPUT_WIDTH * 0.98

        if h_px > 0:
            if h_px >= max_h_px:
                d = 0.0
            elif h_px >= min_h_px:
                d_at_min = (FOCAL_LENGTH * OBJECT_REAL_HEIGHT) / min_h_px
                progress = 1.0 - ((h_px - min_h_px) / (max_h_px - min_h_px))
                d = max(0.0, d_at_min * progress)
            else:
                d = (FOCAL_LENGTH * OBJECT_REAL_HEIGHT) / h_px

        x_center = (x_min + x_max) / 2
        img_center_x = img_shape[1] / 2
        dx_px = x_center - img_center_x
        dx_cm = dx_px * PIXEL_TO_CM * d if d > 0 else 0
        angle = math.degrees(math.atan2(dx_cm, d if d != 0 else 1e-6))

        return d, angle

    def calculate_z_deviation(self, bbox, img_shape, delta=20):
        x_min, y_min, x_max, y_max = bbox
        y_center = (y_min + y_max) / 2
        img_center_y = img_shape[0] / 2
        dy_px = y_center - img_center_y

        if dy_px > delta:
            direction = "down"
        elif dy_px < -delta:
            direction = "up"
        else:
            direction = "center"

        return direction, int(dy_px)


    def start_auto_searching(self):
        """Hàm mới để kích hoạt chế độ tìm kiếm tự động từ GUI."""
        if not self.model_running:
            self.model_running = True
            self.change_state("SEARCHING")
            self.log("[Auto-Search] Bắt đầu quy trình tìm kiếm tự động.", 'info')
        else:
            self.log("[Auto-Search] AI đã chạy, không cần bắt đầu lại.", 'warning')


    def start(self):
        """Hàm start cũ, giờ chỉ bật cờ và chờ phát hiện."""
        self.model_running = True
        self.change_state("TRACKING") # Mặc định vào chế độ bám đuổi
        self.log("[AI Mode] AI Detection Started (Chế độ bám đuổi thủ công).")

    def stop(self):
        self.model_running = False
        self.change_state("IDLE") # Chuyển về trạng thái nghỉ
        self.log("[AI Mode] AI Detection Stopped.")

    def change_state(self, new_state):
        if self.state == new_state:
            return

        # Ghi lại trạng thái cũ TRƯỚC KHI thay đổi
        old_state = self.state 
        self.log(f"Chuyển trạng thái: {old_state} -> {new_state}", 'warning')
        self.state = new_state

        # Logic khi BẮT ĐẦU một trạng thái mới
        if new_state == "IDLE":
            self.serial_main.send("stop")
            self.depth_hold_enabled = False
        
        elif new_state == "SEARCHING":
            # --- LOGIC MỚI: KIỂM TRA TRẠNG THÁI CŨ ---
            # Nếu quay lại từ một sự gián đoạn (mất dấu vật thể)
            if old_state in ["ACQUIRING_TARGET", "TRACKING"]:
                self.log("SEARCH: Tiếp tục mẫu tìm kiếm bị gián đoạn.", 'warning')
                # Quan trọng: Không làm gì cả, giữ nguyên search_step_index, 
                # search_target_yaw và search_state_timer để nó tiếp tục công việc cũ.
                self.depth_hold_enabled = True # Chỉ cần đảm bảo vẫn giữ độ sâu
            
            # Nếu bắt đầu một phiên tìm kiếm mới (từ trạng thái IDLE)
            else:
                self.log("SEARCH: Bắt đầu mẫu tìm kiếm mới từ đầu.", 'info')
                # Reset toàn bộ tiến trình tìm kiếm
                self.search_phase = "SPIRAL"
                self.search_step_index = 0
                self.search_target_yaw = None
                self.search_state_timer = 0

                if self.serial_main.depth_data:
                    self.target_depth = self.serial_main.depth_data[-1]
                    self.depth_hold_enabled = True
                    self.log(f"Bắt đầu giữ độ sâu tại {self.target_depth:.2f}m")
                else:
                    self.log("Không có dữ liệu độ sâu để bắt đầu giữ.", 'error')
                    self.depth_hold_enabled = False
            # ----------------------------------------------------
        
        elif new_state == "ACQUIRING_TARGET":
            self.acquisition_timer = time.monotonic()
            self.serial_main.send("pause")
        
        elif new_state == "TRACKING":
             self.depth_hold_enabled = True

    

    def maintain_depth(self):
         kp_depth, ki_depth, kd_depth = 4.0, 0.0, 0.0
         if self.depth_hold_enabled:
             self.serial_main.send(f"setdepthpid:{kp_depth} {ki_depth} {kd_depth}")
         else:
            self.log("Không giữ độ sâu")

             
         
    def execute_search_pattern(self):
         """Thực hiện mẫu tìm kiếm xoắn ốc rồi đến lưới."""
         pattern = self.SPIRAL_PATTERN if self.search_phase == "SPIRAL" else self.GRID_PATTERN
        
         if self.search_step_index >= len(pattern):
             if self.search_phase == "SPIRAL":
                 self.log("Tìm kiếm xoắn ốc hoàn tất, chuyển sang quét lưới.")
                 self.search_phase = "GRID"
                 self.search_step_index = 0
             else:
                 self.log("Toàn bộ mẫu tìm kiếm hoàn tất. Không tìm thấy đối tượng.", 'error')
                 self.change_state("IDLE")
             return

         action, value = pattern[self.search_step_index]
         current_time = time.monotonic()
        
         if action == 'turn':
             if self.search_target_yaw is None: # Bắt đầu xoay
                 if not self.serial_main.yaw_data:
                     self.log("Thiếu dữ liệu IMU để xoay.", 'error')
                     self.search_step_index += 1
                     return
                 current_yaw = self.serial_main.roll_data[-1]
                 self.search_target_yaw = (current_yaw + value) % 360
                 #self.serial_main.send(f"forward:{self.pwm_base} angle:{value}")
                 if value > 0:
                     self.serial_main.send("turn_left")
                 else:
                     self.serial_main.send("turn_right")
                 self.log(f"SEARCH: Bắt đầu xoay tới {self.search_target_yaw:.1f}°")

             # Kiểm tra xem đã xoay xong chưa
             current_yaw = self.serial_main.roll_data[-1]
             # Tính sai số góc nhỏ nhất
             angle_diff = (self.search_target_yaw - current_yaw + 180) % 360 - 180
             if abs(angle_diff) < 10: # Ngưỡng 10 độ
                 self.log("SEARCH: Xoay hoàn tất.")
                 self.search_target_yaw = None
                 self.search_step_index += 1
        
         elif action == 'forward':
             if self.search_state_timer == 0: # Bắt đầu đi thẳng
                 self.serial_main.send(f"forward:1700 angle:0.0")
                 self.search_state_timer = current_time
                 self.log(f"SEARCH: Đi thẳng trong {value} giây.")

             if current_time - self.search_state_timer > value:
                 self.log("SEARCH: Đi thẳng hoàn tất.")
                 self.search_state_timer = 0
                 self.search_step_index += 1
    
    
    def execute_target_acquisition(self, boxes):
        """Thực hiện hành động phản xạ để khóa mục tiêu."""
        # **HÀNH ĐỘNG PHẢN XẠ 2: GỬI LỆNH HIỆU CHỈNH**
        self.serial_main.send(f"forward:{self.pwm_base} angle:{self.last_known_angle:.1f}")
        self.log(f"ACQUIRING: Gửi lệnh hiệu chỉnh góc {self.last_known_angle:.1f}°")

        # Kiểm tra xem đã khóa được chưa hoặc hết thời gian
        if time.monotonic() - self.acquisition_timer > self.ACQUISITION_TIMEOUT:
            self.log("ACQUIRING: Hết thời gian chờ. Quay lại tìm kiếm.", 'error')
            self.change_state("SEARCHING")
            return

        if boxes: # Nếu vẫn thấy đối tượng trong các frame tiếp theo
            _, current_angle = self.calculate_distance_and_angle(boxes[0].xyxy[0].tolist(), (480,480))
            if abs(current_angle) < 10.0: # Ngưỡng 5 độ để xác nhận khóa
                self.log("ACQUIRING: Khóa mục tiêu thành công!", 'info')
                self.change_state("TRACKING")

    def execute_tracking(self, boxes):
        """Logic bám đuổi mục tiêu khi đã khóa."""
        d, angle = self.calculate_distance_and_angle(boxes[0].xyxy[0].tolist(), (480,480))
        label = f"TRACKING: {d:.1f}cm, {angle:.1f}°"
        self.log(label)
            
        if d > 40: # Giữ khoảng cách 30cm
            self.serial_main.send(f"forward:{self.pwm_base} angle:{angle:.1f}")
        else:
            self.serial_main.send("pause")

    def run_loop(self):
        """Vòng lặp chính của AI, điều khiển bởi máy trạng thái."""
        while True:
            try:
                # 1. Lấy frame từ camera (định dạng gốc là RGB)
                frame_rgb = self.picam2.capture_array()

                # 2. SỬA LỖI: Chuẩn hóa định dạng màu sắc sang BGR ngay từ đầu.
                #    Tất cả các bước xử lý sau sẽ dùng frame BGR này.
                #processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                processed_frame = frame_rgb
                
                # --- MÁY TRẠNG THÁI (STATE MACHINE) ---
                if not self.model_running:
                    self.change_state("IDLE")
                else: # AI đang chạy
                    # Chạy mô hình YOLO trên frame BGR
                    results = self.yolo_model(processed_frame, imgsz=IMAGE_INPUT_WIDTH, conf=0.6, verbose=False)
                    result = results[0]
                    boxes = result.boxes

                    # SỬA LỖI: Thống nhất tên biến
                    object_detected = bool(boxes)

                    # Vẽ bounding box và label tùy chỉnh
                    if object_detected:
                        # result.plot() sẽ vẽ lên frame và trả về một frame BGR mới
                        annotated_frame = result.plot()
                        
                        # Vòng lặp để lấy thông tin và vẽ thêm label tùy chỉnh
                        for box in boxes:
                            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                            cls_id = int(box.cls.cpu().numpy())
                            bbox = [x_min, y_min, x_max, y_max]

                            d, angle = self.calculate_distance_and_angle(bbox, processed_frame.shape)
                            direction_z, dy_px = self.calculate_z_deviation(bbox, processed_frame.shape, delta=20)

                            label = f"{self.yolo_model.names[cls_id]}: {d:.1f}cm, {angle:.1f}d, dy={dy_px:+d}"
                            
                            # Vẽ thông tin tùy chỉnh lên frame đã được YOLO vẽ sẵn
                            cv2.putText(annotated_frame, label, (int(x_min), int(y_min) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        # Cập nhật frame để hiển thị là frame đã được vẽ
                        processed_frame = annotated_frame
                    
                    # **CƠ CHẾ NGẮT (INTERRUPT REFLEX)** - Giờ đã dùng biến 'object_detected' đúng
                    if object_detected and self.state == "SEARCHING":
                        self.last_known_distance, self.last_known_angle = self.calculate_distance_and_angle(boxes[0].xyxy[0].tolist(), processed_frame.shape)
                        self.change_state("ACQUIRING_TARGET")
                    
                    # **CÁC TRẠNG THÁI CHÍNH**
                    if self.state == "IDLE":
                        self.change_state("SEARCHING")
                    elif self.state == "SEARCHING":
                        self.execute_search_pattern()
                        self.maintain_depth()
                    elif self.state == "ACQUIRING_TARGET":
                        self.execute_target_acquisition(boxes)
                    elif self.state == "TRACKING":
                        if not object_detected:
                            self.log("Mất dấu mục tiêu, quay lại tìm kiếm.", 'warning')
                            self.change_state("SEARCHING")
                        else:
                            self.execute_tracking(boxes)
                            self.maintain_depth()
                
                # --- Xử lý hiển thị (FPS và gửi tới GUI) ---
                # Vẽ FPS lên frame (lúc này luôn là BGR)
                processed_frame = self.draw_fps_on_frame(processed_frame)

                # SỬA LỖI: Chuyển đổi từ BGR sang RGB ở bước cuối cùng, đảm bảo luôn đúng
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                try:
                    # Bỏ frame cũ đi nếu có và thêm frame mới vào
                    if not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(img_tk)
                except queue.Full:
                    pass

            except Exception as e:
                self.log(f"[AI Error] {e}", 'error')
                time.sleep(1)

   
    def draw_fps_on_frame(self, frame):
        self.frame_counter += 1
        now = time.monotonic()
        if now - self.fps_timer >= self.fps_update_interval:
            self.fps = self.frame_counter / (now - self.fps_timer)
            self.fps_timer = now
            self.frame_counter = 0

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame
    

   

class GuiMain:
    def __init__(self, serial_main, ai_main):
        self.serial_main = serial_main
        self.ai_main = ai_main

        self.root = tk.Tk()
        self.root.title("ROV Control Panel")
        self.root.geometry("1200x900")

        self.gui_control_frame = tk.Frame(self.root)
        self.gui_ai_frame = tk.Frame(self.root)

        self.build_gui_control()
        self.build_gui_ai()

        # Thay vì chỉ thiết lập một widget, ta sẽ truyền cả hai để AI class có thể log
        # Việc log thực tế sẽ được xử lý trong process_serial_log
        self.ai_main.set_pwm_entry_widget(self.ai_entry_forward_pwm)


        self.show_gui_control()
        self.process_serial_log() # Bắt đầu xử lý log
        self.update_loop() # Bắt đầu cập nhật hình ảnh
        self.root.mainloop()


    def process_serial_log(self):
        """Xử lý các tin nhắn log từ queue và cập nhật CẢ HAI widget log."""
        while not self.serial_main.log_queue.empty():
            try:
                msg = self.serial_main.log_queue.get_nowait()
                
                # Xác định tag dựa trên nội dung tin nhắn
                tag = 'info'
                if "[Error]" in msg or "Lỗi" in msg or "error" in msg.lower(): tag = 'error'
                elif "[Warning]" in msg or "warning" in msg.lower(): tag = 'warning'

                # Cập nhật widget log của màn hình control
                self.log_text_widget.config(state='normal')
                self.log_text_widget.insert(tk.END, msg + "\n", tag)
                self.log_text_widget.see(tk.END)
                self.log_text_widget.config(state='disabled')
                
                # Cập nhật widget log của màn hình AI
                if hasattr(self, 'ai_log_text_widget') and self.ai_log_text_widget:
                    self.ai_log_text_widget.config(state='normal')
                    self.ai_log_text_widget.insert(tk.END, msg + "\n", tag)
                    self.ai_log_text_widget.see(tk.END)
                    self.ai_log_text_widget.config(state='disabled')

            except queue.Empty:
                pass
        self.root.after(100, self.process_serial_log)

    def send_dir(self):
        try:
            values = [int(e.get()) for e in self.dir_entries]
            self.serial_main.send(f"dir:{' '.join(map(str, values))}")
            self.ai_main.log(f"[GUI Control] Sent DIR: {' '.join(map(str, values))}", 'info')
        except ValueError:
            self.ai_main.log("[Lỗi] Nhập số nguyên hợp lệ cho DIR\n", 'error')

    def send_dive(self):
        try:
            values = [int(e.get()) for e in self.dive_entries]
            self.serial_main.send(f"dive:{' '.join(map(str, values))}")
            self.ai_main.log(f"[GUI Control] Sent DIVE: {' '.join(map(str, values))}", 'info')
        except ValueError:
            self.ai_main.log("[Lỗi] Vui lòng nhập số nguyên hợp lệ cho DIVE", 'error')
        except Exception as e:
            self.ai_main.log(f"[Serial Error] Không thể gửi DIVE: {e}", 'error')
            
    def send_dive_command(self):
        """Gửi lệnh LẶN sử dụng hàm send_pid_roll."""
        self.send_pid_roll(pwm1=1700, pwm2=1700)
        self.ai_main.log("[GUI Control] Sent DIVE command (PWM: 1700, 1700).", 'info')

    def send_surface_command(self):
        """Gửi lệnh NỔI sử dụng hàm send_pid_roll."""
        self.send_pid_roll(pwm1=1200, pwm2=1200)
        self.ai_main.log("[GUI Control] Sent SURFACE command (PWM: 1200, 1200).", 'info')

    def send_stop(self):
        self.serial_main.send("stop")
        self.ai_main.log("[GUI Control] Sent STOP command.", 'info')

    def send_calibrate_command(self):
        self.serial_main.send("calib")
        self.ai_main.log("[GUI Control] Sent CALIBRATE command.", 'info')


    def send_turn_left_command(self):
        self.serial_main.send("turn_left")
        self.ai_main.log("[GUI Control] Sent TURN_LEFT command.", 'info')

    def send_turn_right_command(self):
        self.serial_main.send("turn_right")
        self.ai_main.log("[GUI Control] Sent TURN_RIGHT command.", 'info')

    #PID YAW
    def send_pid_values(self):
        try:
            kp_yaw, ki_yaw, kd_yaw = 4.3, 0.2, 0.1
            pwm = int(self.entry_forward_pwm.get())
            self.serial_main.send(f"setpid:{kp_yaw} {ki_yaw} {kd_yaw}")
            self.serial_main.send(f"forward:{pwm} angle:0.0")
            self.ai_main.log(f"[GUI Control] Sent PID Roll: Kp={kp_yaw}, Ki={ki_yaw}, Kd={kd_yaw}, PWM Base={pwm}", 'info')
        except ValueError:
            self.ai_main.log("[Lỗi] Nhập PID hoặc PWM hợp lệ", 'error')

    #PID ROLL = 0
    def send_pid_roll(self,pwm1,pwm2):
        try:
            kp_roll, ki_roll, kd_roll = 4.0, 0.0, 0.0
            self.serial_main.send(f"pidyaw:{kp_roll} {ki_roll} {kd_roll} {pwm1} {pwm2}")
            self.ai_main.log(f"[GUI Control] Sent PID Yaw: Kp={kp_roll}, Ki={ki_roll}, Kd={kd_roll}, PWM1={pwm1}, PWM2={pwm2}", 'info')
        except ValueError:
            self.ai_main.log("[Lỗi] Nhập PID Yaw và PWM hợp lệ", 'error')

    #PID DEPTH BALANCE
    def send_depth_pid_values(self):
        try:
            kp_depth, ki_depth, kd_depth = 4.0, 0.0, 0.0
            self.serial_main.send(f"setdepthpid:{kp_depth} {ki_depth} {kd_depth}")
            self.ai_main.log(f"[GUI Control] Sent PID Depth: Kp={kp_depth}, Ki={ki_depth}, Kd={kd_depth}", 'info')
        except ValueError:
            self.ai_main.log("[Lỗi] Nhập PID Depth hợp lệ", 'error')
    
    def build_gui_control(self):
        self.gui_control_frame.pack(fill=tk.BOTH, expand=True)

        main_frame = tk.Frame(self.gui_control_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame, bd=2, relief="groove", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Chuyển sang chế độ AI
        tk.Button(control_frame, text="Chuyển sang chế độ AI", command=self.show_gui_ai, bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Điều khiển DIR
        tk.Label(control_frame, text="Động cơ điều hướng (1 3 4 6):", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        dir_entries_frame = tk.Frame(control_frame)
        dir_entries_frame.pack()
        self.dir_entries = [tk.Entry(dir_entries_frame, width=6) for _ in range(4)]
        for e in self.dir_entries:
            e.pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Gửi DIR", command=self.send_dir, bg="#4CAF50", fg="white", font=("Arial", 10)).pack(pady=5)

        # Điều khiển DIVE (2 động cơ lặn)
        tk.Label(control_frame, text="Động cơ lặn (2 5):", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        dive_entries_frame = tk.Frame(control_frame)
        dive_entries_frame.pack()
        self.dive_entries = [tk.Entry(dive_entries_frame, width=6) for _ in range(2)]
        for e in self.dive_entries:
            e.pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Gửi DIVE", command=self.send_dive, bg="#2196F3", fg="white", font=("Arial", 10)).pack(pady=5)

        tk.Button(control_frame, text="STOP", command=self.send_stop, fg="white", bg="red", font=("Arial", 10, "bold")).pack(pady=10)
        tk.Button(control_frame, text="Hiệu chỉnh IMU (0°)", command=self.send_calibrate_command, bg="#FFC107", fg="black", font=("Arial", 10)).pack(pady=10)
        
        # Điều khiển rẽ trái/phải
        turn_buttons_frame = tk.Frame(control_frame)
        turn_buttons_frame.pack(pady=5)
        tk.Button(turn_buttons_frame, text="Rẽ trái", command=self.send_turn_left_command, bg="#795548", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(turn_buttons_frame, text="Rẽ phải", command=self.send_turn_right_command, bg="#795548", fg="white", font=("Arial", 10)).pack(side=tk.RIGHT, padx=5)

        # PID YAW
        # PWM cơ bản
        tk.Label(control_frame, text="PWM cơ bản khi tiến (ví dụ: 1550)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.entry_forward_pwm = tk.Entry(control_frame, width=10)
        self.entry_forward_pwm.pack()
        tk.Button(control_frame, text="Gửi PID & Tiến", command=self.send_pid_values, bg="#8BC34A", fg="white", font=("Arial", 10)).pack(pady=10)

        # PID ROLL = 0 
        frame_dive_control = tk.LabelFrame(control_frame, text="Điều khiển Lặn / Nổi (PID)", font=("Arial", 10, "bold"), padx=10, pady=5)
        frame_dive_control.pack(pady=10, fill=tk.X)
        buttons_frame = tk.Frame(frame_dive_control)
        buttons_frame.pack(pady=5)
        tk.Button(buttons_frame, text="Lặn", command=self.send_dive_command, bg="#2196F3", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10, ipadx=10)
        tk.Button(buttons_frame, text="Nổi", command=self.send_surface_command, bg="#FF9800", fg="white", font=("Arial", 10)).pack(side=tk.RIGHT, padx=10, ipadx=10)

        # PID Depth
        frame_depth_pid = tk.LabelFrame(control_frame, text="PID Độ sâu", font=("Arial", 10, "bold"))
        frame_depth_pid.pack(pady=10, fill=tk.X)
        tk.Button(frame_depth_pid, text="Gửi PID Depth", command=self.send_depth_pid_values, bg="#9C27B0", fg="white", font=("Arial", 10)).grid(row=0, column=8, padx=10, pady=5)
        
        # Log hoạt động
        log_frame = tk.LabelFrame(control_frame, text="Log Hoạt Động", font=("Arial", 10, "bold"))
        log_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.log_text_widget = tk.Text(log_frame, height=10, width=50, state='disabled', wrap='word', bg='#f0f0f0')
        self.log_text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.log_text_widget.tag_configure('error', foreground='red', font=('Arial', 9, 'bold'))
        self.log_text_widget.tag_configure('info', foreground='blue')
        self.log_text_widget.tag_configure('warning', foreground='orange')

        self.build_plot_area(main_frame)

    def build_gui_ai(self):
        # Khung chứa nút điều khiển AI (bên trái)
        ai_control_frame = tk.Frame(self.gui_ai_frame, bd=2, relief="groove", padx=10, pady=10)
        ai_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # --- Khung bên phải chứa Camera và Log ---
        right_pane = tk.Frame(self.gui_ai_frame)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Khung hiển thị camera (video feed) ở trên
        self.ai_image_label = tk.Label(right_pane, borderwidth=2, relief="solid")
        self.ai_image_label.pack(fill=tk.BOTH, expand=True)
        
        # --- BỔ SUNG: Khung log cho màn hình AI ở dưới ---
        ai_log_frame = tk.LabelFrame(right_pane, text="Log Hoạt Động AI", font=("Arial", 10, "bold"))
        ai_log_frame.pack(fill=tk.X, expand=False, pady=(10,0)) # Không expand theo chiều dọc
        
        self.ai_log_text_widget = tk.Text(ai_log_frame, height=12, state='disabled', wrap='word', bg='#f0f0f0')
        self.ai_log_text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.ai_log_text_widget.tag_configure('error', foreground='red', font=('Arial', 9, 'bold'))
        self.ai_log_text_widget.tag_configure('info', foreground='blue')
        self.ai_log_text_widget.tag_configure('warning', foreground='orange')

        # --- Các nút trong ai_control_frame (bên trái) ---
        tk.Button(ai_control_frame, text="Start Auto-Searching", command=self.ai_main.start_auto_searching,
                  bg="#9C27B0", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=10)
        tk.Button(ai_control_frame, text="Start Manual Tracking", command=self.ai_main.start,
                    bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=10)

        tk.Button(ai_control_frame, text="Stop AI", command=self.ai_main.stop,
                    bg="red", fg="white", font=("Arial", 12, "bold"), width=20).pack(pady=10)

        tk.Button(ai_control_frame, text="Quay về điều khiển thủ công", command=self.show_gui_control,
                    bg="#607D8B", fg="white", font=("Arial", 12), width=20).pack(pady=20)
        
        # PWM cơ bản
        tk.Label(ai_control_frame, text="PWM cơ bản khi tiến (AI)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.ai_entry_forward_pwm = tk.Entry(ai_control_frame, width=10)
        self.ai_entry_forward_pwm.pack()
        self.ai_entry_forward_pwm.insert(0, "1550") # Giá trị mặc định
        
        tk.Button(ai_control_frame, text="STOP (AI)", command=self.ai_main.send_stop, fg="white", bg="red", font=("Arial", 10, "bold")).pack(pady=10)


    def build_plot_area(self, parent):
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_imu = Figure(figsize=(6, 4), dpi=100)
        self.ax_imu = self.fig_imu.add_subplot(111)
        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=plot_frame)
        self.canvas_imu.get_tk_widget().pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.ani_imu = FuncAnimation(self.fig_imu, self.update_imu_plot, interval=500)

        self.fig_pressure = Figure(figsize=(6, 4), dpi=100)
        self.ax_pressure = self.fig_pressure.add_subplot(111)
        self.canvas_pressure = FigureCanvasTkAgg(self.fig_pressure, master=plot_frame)
        self.canvas_pressure.get_tk_widget().pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.ani_pressure = FuncAnimation(self.fig_pressure, self.update_pressure_plot, interval=500)

    def show_gui_control(self):
        self.ai_main.stop() # Tắt AI khi chuyển về chế độ điều khiển thủ công
        self.gui_ai_frame.pack_forget()
        self.gui_control_frame.pack(fill=tk.BOTH, expand=True)

    def show_gui_ai(self):
        self.gui_control_frame.pack_forget()
        self.gui_ai_frame.pack(fill=tk.BOTH, expand=True)

    def update_loop(self):
        # Kiểm tra nếu cửa sổ AI đang hiển thị thì mới cập nhật hình ảnh
        if self.gui_ai_frame.winfo_ismapped():
            self.update_image()
        self.root.after(30, self.update_loop)

    def update_image(self):
        # Luồng GUI giờ chỉ có nhiệm vụ lấy và hiển thị, không xử lý nặng.
        try:
            img_tk = self.ai_main.frame_queue.get_nowait()
            if hasattr(self, 'ai_image_label'):
                # Giữ tham chiếu để ảnh không bị "dọn rác"
                self.ai_image_label.imgtk = img_tk
                self.ai_image_label.config(image=img_tk)
        except queue.Empty:
            # Nếu queue rỗng (chưa có frame mới), không làm gì cả.
            pass

    def update_imu_plot(self, i):
        self.ax_imu.clear()
        self.ax_imu.plot(self.serial_main.imu_time_data, self.serial_main.roll_data, label="Yaw", color='red')
        self.ax_imu.plot(self.serial_main.imu_time_data, self.serial_main.pitch_data, label="Pitch", color='green')
        self.ax_imu.plot(self.serial_main.imu_time_data, self.serial_main.yaw_data, label="Roll", color='blue')
        self.ax_imu.set_title("Góc Roll / Pitch / Yaw (IMU)", fontsize=12)
        self.ax_imu.set_xlabel("Thời gian (s)", fontsize=10)
        self.ax_imu.set_ylabel("Góc (°)", fontsize=10)
        self.ax_imu.grid(True)
        self.ax_imu.legend(loc='upper right', fontsize=8)
        self.fig_imu.tight_layout()

    def update_pressure_plot(self, i):
        self.ax_pressure.clear()
        self.ax_pressure.plot(self.serial_main.pressure_time_data, self.serial_main.pressure_data, label="Độ sâu Setpoint", color='purple', marker = 'o')
        self.ax_pressure.plot(self.serial_main.pressure_time_data, self.serial_main.depth_data, label="Độ sâu (m)", color='red')
        self.ax_pressure.set_title("Độ sâu ROV", fontsize=12)
        self.ax_pressure.set_xlabel("Thời gian (s)", fontsize=10)
        self.ax_pressure.set_ylabel("Độ sâu (m)", fontsize=10)
        self.ax_pressure.grid(True)
        self.ax_pressure.legend(loc='upper right', fontsize=8)
        self.fig_pressure.tight_layout()


if __name__ == "__main__":
    serial_main = SerialMain()
    ai_main = AiMain(serial_main) # Truyền serial_main vào ai_main
    GuiMain(serial_main, ai_main)
