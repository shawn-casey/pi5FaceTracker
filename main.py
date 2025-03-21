import cv2
import time
import math
import threading
import numpy as np
import os
from sshkeyboard import listen_keyboard, stop_listening
from adafruit_servokit import ServoKit

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
RTSP_OUTPUT = 'rtsp://localhost:8888/cam'

PCA9685_ADDRESS = 0x40  # Default address
PCA9685_FREQUENCY = 50  # Standard frequency for standard servos

SERVO_PAN_CHANNEL = 0
SERVO_TILT_CHANNEL = 1
PAN_MIN_ANGLE = 0
PAN_MAX_ANGLE = 180
TILT_MIN_ANGLE = 0
TILT_MAX_ANGLE = 90
PAN_CENTER = 90
TILT_CENTER = 45

KP = 0.1  # Proportional gain
KD = 0.05  # Derivative gain
SMOOTHING_FACTOR = 0.3  # For exponential smoothing of movements

pan_angle = PAN_CENTER
tilt_angle = TILT_CENTER
last_error_x = 0
last_error_y = 0
auto_tracking = True
face_detected = False
last_face_time = 0
FACE_TIMEOUT = 2.0  # Seconds to wait after losing face before returning to center

class TrackerCamera:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Could not load Haar Cascade from {HAAR_CASCADE_PATH}")
        
        # Initialize servo controller
        print("Initializing servo controller...")
        self.servo_kit = ServoKit(channels=16, address=PCA9685_ADDRESS)
        self.servo_kit.servo[SERVO_PAN_CHANNEL].set_pulse_width_range(500, 2500)  # Adjust based on your servos
        self.servo_kit.servo[SERVO_TILT_CHANNEL].set_pulse_width_range(500, 2500)
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)  # Use 0 for the first camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Initialize RTSP streaming
        print("Setting up RTSP stream...")
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG: Motion JPG
        self.out = cv2.VideoWriter(
            f'appsrc ! videoconvert ! video/x-raw,format=I420 ! '
            f'x264enc speed-preset=ultrafast tune=zerolatency ! '
            f'rtspclientsink location={RTSP_OUTPUT}',
            cv2.CAP_GSTREAMER, 0, CAMERA_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT), True
        )
        
        # Center servos at startup
        self.center_servos()
        print("Initialization complete")
        
    def center_servos(self):
        global pan_angle, tilt_angle
        pan_angle = PAN_CENTER
        tilt_angle = TILT_CENTER
        self.set_servo_angles(pan_angle, tilt_angle)
        print(f"Servos centered at Pan: {pan_angle}, Tilt: {tilt_angle}")
    
    def set_servo_angles(self, pan, tilt):
        # Clamp values to valid ranges
        pan = max(PAN_MIN_ANGLE, min(PAN_MAX_ANGLE, pan))
        tilt = max(TILT_MIN_ANGLE, min(TILT_MAX_ANGLE, tilt))
        
        # Set servo positions
        self.servo_kit.servo[SERVO_PAN_CHANNEL].angle = pan
        self.servo_kit.servo[SERVO_TILT_CHANNEL].angle = tilt
        
    def detect_faces(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def track_face(self, frame):
        global pan_angle, tilt_angle, last_error_x, last_error_y, face_detected, last_face_time
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # If faces were found, track the largest one --> closest face
        if len(faces) > 0:
            face_detected = True
            last_face_time = time.time()
            
            # Find the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Draw circle at face center
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
            
            # Calculate error (distance from frame center)
            error_x = center_x - face_center_x
            error_y = face_center_y - center_y  # Inverted for tilt
            
            # PID calculation (simplified P-D control)
            p_term_x = KP * error_x
            d_term_x = KD * (error_x - last_error_x)
            p_term_y = KP * error_y
            d_term_y = KD * (error_y - last_error_y)
            
            # Update last error values
            last_error_x = error_x
            last_error_y = error_y
            
            # Apply smoothing to avoid jerky movements
            delta_pan = p_term_x + d_term_x
            delta_tilt = p_term_y + d_term_y
            
            # Update servo angles with smoothing
            pan_angle += delta_pan * SMOOTHING_FACTOR
            tilt_angle += delta_tilt * SMOOTHING_FACTOR
            
            # Set servo positions if in auto tracking mode
            if auto_tracking:
                self.set_servo_angles(pan_angle, tilt_angle)
        else:
            # No face detected
            face_detected = False
            
            # Return to center after timeout
            if not face_detected and (time.time() - last_face_time) > FACE_TIMEOUT:
                # Gradually return to center
                pan_diff = PAN_CENTER - pan_angle
                tilt_diff = TILT_CENTER - tilt_angle
                
                if abs(pan_diff) > 1 or abs(tilt_diff) > 1:
                    pan_angle += pan_diff * 0.1
                    tilt_angle += tilt_diff * 0.1
                    
                    if auto_tracking:
                        self.set_servo_angles(pan_angle, tilt_angle)
        
        # Draw crosshair at center of frame
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (frame_width, center_y), (255, 0, 0), 1)
        
        # Add status text
        tracking_status = "AUTO" if auto_tracking else "MANUAL"
        face_status = "TRACKING" if face_detected else "SEARCHING"
        cv2.putText(frame, f"Mode: {tracking_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {face_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Pan: {int(pan_angle)} Tilt: {int(tilt_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def process_keyboard_input(self, key):
        global pan_angle, tilt_angle, auto_tracking
        
        # Manual control keys
        if key == "a":  # Pan left
            pan_angle += 5
            self.set_servo_angles(pan_angle, tilt_angle)
        elif key == "d":  # Pan right
            pan_angle -= 5
            self.set_servo_angles(pan_angle, tilt_angle)
        elif key == "w":  # Tilt up
            tilt_angle -= 5
            self.set_servo_angles(pan_angle, tilt_angle)
        elif key == "s":  # Tilt down
            tilt_angle += 5
            self.set_servo_angles(pan_angle, tilt_angle)
        elif key == "c":  # Center
            self.center_servos()
        elif key == "t":  # Toggle tracking mode
            auto_tracking = not auto_tracking
            print(f"Auto tracking: {'ON' if auto_tracking else 'OFF'}")
        elif key == "q":  # Quit
            self.cleanup()
            return True  # Stop
        
        return False  # Continue running
    
    def keyboard_thread(self):
        # Start keyboard listener in a separate thread
        listen_keyboard(
            on_press=self.process_keyboard_input,
            delay_second_char=0.1
        )
    
    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Start keyboard thread
        kb_thread = threading.Thread(target=self.keyboard_thread)
        kb_thread.daemon = True
        kb_thread.start()
        
        print("Starting camera feed. Press 'q' to quit.")
        print("Controls: w/a/s/d - pan/tilt, c - center, t - toggle tracking")
        
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Mirror image for more intuitive control
            frame = cv2.flip(frame, 1)
            
            # Process frame for face tracking
            processed_frame = self.track_face(frame)
            
            # Write to RTSP stream
            if self.out.isOpened():
                self.out.write(processed_frame)
            
            # Display frame locally
            cv2.imshow('Smart Tracking Camera', processed_frame)
            
            # Check for keyboard interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        print("Cleaning up...")
        stop_listening()
        self.center_servos()
        time.sleep(0.5)  # Allow servos to reach center
        
        if self.cap.isOpened():
            self.cap.release()
        
        if self.out.isOpened():
            self.out.release()
            
        cv2.destroyAllWindows()
        print("Shutdown complete")

if __name__ == "__main__":
    try:
        tracker = TrackerCamera()
        tracker.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")