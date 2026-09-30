"""
Hand detection and landmark extraction using MediaPipe Tasks API
"""
import cv2
import numpy as np
import os
import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError as e:
    MP_AVAILABLE = False
    raise ImportError(
        f"MediaPipe import failed: {e}\n"
        "Please ensure MediaPipe is installed: pip install mediapipe"
    )
except Exception as e:
    MP_AVAILABLE = False
    raise ImportError(
        f"MediaPipe initialization error: {e}\n"
        "Please ensure MediaPipe is installed correctly: pip install --upgrade mediapipe"
    )

def download_hand_landmarker_model():
    """Download the hand landmarker model if it doesn't exist"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'hand_landmarker.task')
    
    if not os.path.exists(model_path):
        print("Downloading hand landmarker model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}\n"
                "Please download manually from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                f"Save it to: {model_path}"
            )
    
    return model_path

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Hand Landmarker using Tasks API
        """
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Please install it with: pip install mediapipe")
        
        # Download model if needed
        model_path = download_hand_landmarker_model()
        
        # Create base options for hand landmarker with model path
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Create hand landmarker options
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Create hand landmarker
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Store drawing utilities (for compatibility)
        self.mp_drawing = None
        
    def detect_hands(self, image):
        """
        Detect hands in the image and return results
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            results: HandLandmarkerResult object
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect hands
        results = self.detector.detect(mp_image)
        
        return results
    
    def extract_landmarks(self, results, image_shape):
        """
        Extract normalized hand landmarks as a feature vector
        
        Args:
            results: HandLandmarkerResult object
            image_shape: Tuple of (height, width)
            
        Returns:
            landmarks: List of landmark arrays (one per hand) or None
        """
        if not results.hand_landmarks:
            return None
            
        landmarks_list = []
        for hand_landmarks in results.hand_landmarks:
            # Extract 21 landmarks (x, y, z) = 63 features
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(np.array(landmarks))
            
        return landmarks_list
    
    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks on the image
        
        Args:
            image: BGR image
            results: HandLandmarkerResult object
            
        Returns:
            image: Image with landmarks drawn
        """
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                # Draw connections (simplified - drawing main connections)
                if len(hand_landmarks) >= 21:
                    # Thumb
                    self._draw_line(image, hand_landmarks, [0, 1, 2, 3, 4], (255, 0, 0))
                    # Index finger
                    self._draw_line(image, hand_landmarks, [0, 5, 6, 7, 8], (255, 0, 0))
                    # Middle finger
                    self._draw_line(image, hand_landmarks, [0, 9, 10, 11, 12], (255, 0, 0))
                    # Ring finger
                    self._draw_line(image, hand_landmarks, [0, 13, 14, 15, 16], (255, 0, 0))
                    # Pinky
                    self._draw_line(image, hand_landmarks, [0, 17, 18, 19, 20], (255, 0, 0))
                    # Palm
                    self._draw_line(image, hand_landmarks, [5, 9, 13, 17, 0], (255, 0, 0))
        
        return image
    
    def _draw_line(self, image, landmarks, indices, color):
        """Helper method to draw lines between landmarks"""
        h, w = image.shape[:2]
        for i in range(len(indices) - 1):
            pt1 = landmarks[indices[i]]
            pt2 = landmarks[indices[i + 1]]
            x1 = int(pt1.x * w)
            y1 = int(pt1.y * h)
            x2 = int(pt2.x * w)
            y2 = int(pt2.y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
