"""
Main application for Naruto Jutsu Hand Gesture Recognition
"""
import cv2
import numpy as np
import platform
from hand_detector import HandDetector
from gesture_classifier import JutsuRecognizer
from visual_effects import VisualEffects
import time

class NarutoJutsuApp:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.jutsu_recognizer = JutsuRecognizer()
        self.visual_effects = VisualEffects()
        self.cap = None
        self.current_jutsu = None
        self.jutsu_start_time = None
        self.effect_duration = 3.0  # seconds
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        # AVFoundation is the native macOS camera backend. Explicitly using it
        # avoids OpenCV trying an incompatible fallback backend.
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the camera. On macOS, enable Camera access for "
                "the app launching Python (VS Code or Terminal) in "
                "System Settings > Privacy & Security > Camera, then restart it."
            )
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Fail early if the camera opens but cannot deliver frames.
        ok, _ = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError(
                "The camera opened but returned no frames. Check camera permissions "
                "and make sure no other app is using it."
            )
        
    def run(self):
        """Main application loop"""
        self.initialize_camera()
        
        print("Naruto Jutsu Recognition System Started!")
        print("Perform hand gestures to trigger jutsu effects!")
        print("Press 'Q' to quit")
        
        frame_count = 0
        last_prediction_time = time.time()
        prediction_interval = 0.1  # Predict every 100ms
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = self.hand_detector.detect_hands(frame)
            
            # Draw hand landmarks
            frame = self.hand_detector.draw_landmarks(frame, results)
            
            # Extract landmarks and predict gesture
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                landmarks = self.hand_detector.extract_landmarks(results, frame.shape[:2])
                
                if landmarks:
                    jutsu_name, confidence = self.jutsu_recognizer.predict(landmarks)
                    
                    # Trigger effect if confidence is high
                    if confidence > 0.7 and jutsu_name != "None":
                        if self.current_jutsu != jutsu_name:
                            self.current_jutsu = jutsu_name
                            self.jutsu_start_time = current_time
                            print(f"Jutsu detected: {jutsu_name} (Confidence: {confidence:.2f})")
                
                last_prediction_time = current_time
            
            # Apply visual effects if jutsu is active
            if self.current_jutsu and self.jutsu_start_time:
                elapsed_time = current_time - self.jutsu_start_time
                if elapsed_time < self.effect_duration:
                    frame = self.visual_effects.trigger_effect(self.current_jutsu, frame)
                    
                    # Display jutsu name
                    cv2.putText(frame, f"{self.current_jutsu}!", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                              1.5, (0, 255, 255), 3)
                else:
                    # Effect duration expired
                    self.current_jutsu = None
                    self.jutsu_start_time = None
            
            # Display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - last_prediction_time + 0.001)
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                          (50, frame.shape[0] - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Perform hand gestures to trigger jutsu!", 
                      (50, frame.shape[0] - 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Naruto Jutsu Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                # Save screenshot
                cv2.imwrite(f'screenshot_{int(time.time())}.jpg', frame)
                print("Screenshot saved!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    app = NarutoJutsuApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
