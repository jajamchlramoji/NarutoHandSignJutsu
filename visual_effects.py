"""
Visual effects for Naruto jutsu
"""
import cv2
import numpy as np
from typing import Tuple, List
import time

class VisualEffects:
    """
    Manages visual effects triggered by jutsu gestures
    """
    def __init__(self):
        self.active_effects = {}
        self.effect_duration = 3.0  # seconds
        self.clone_count = 10
        
    def trigger_effect(self, jutsu_name: str, frame: np.ndarray):
        """
        Trigger a visual effect based on jutsu name
        
        Args:
            jutsu_name: Name of the jutsu
            frame: Current camera frame
            
        Returns:
            frame: Frame with effect applied
        """
        if jutsu_name == "Cloning Jutsu":
            return self.cloning_jutsu(frame)
        elif jutsu_name == "Fireball Jutsu":
            return self.fireball_jutsu(frame)
        elif jutsu_name == "Water Jutsu":
            return self.water_jutsu(frame)
        elif jutsu_name == "Wind Jutsu":
            return self.wind_jutsu(frame)
        elif jutsu_name == "Rasengan":
            return self.rasengan_jutsu(frame)
        
        return frame
    
    def cloning_jutsu(self, frame: np.ndarray) -> np.ndarray:
        """
        Create multiple clones of the person in the frame
        
        Args:
            frame: Current camera frame
            
        Returns:
            frame: Frame with clones added
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        
        # Calculate center point
        center_x = w // 2
        center_y = h // 2
        
        # Create clones arranged in a circle around the center
        clone_positions = []
        for i in range(self.clone_count):
            # Calculate position in a circle
            angle = (2 * np.pi * i) / self.clone_count
            # Use larger radius to spread clones more
            radius = min(w, h) * 0.25
            
            x_offset = int(radius * np.cos(angle))
            y_offset = int(radius * np.sin(angle))
            
            clone_positions.append((x_offset, y_offset))
        
        # Create clones with varying sizes and transparency
        for i, (x_offset, y_offset) in enumerate(clone_positions):
            # Vary clone size (smaller clones appear further)
            scale = 0.5 + (i % 4) * 0.15  # Range from 0.5 to 0.95
            clone_w = int(w * scale)
            clone_h = int(h * scale)
            
            # Resize the original frame to create clone
            clone = cv2.resize(frame, (clone_w, clone_h))
            
            # Calculate clone position
            x1 = center_x - clone_w // 2 + x_offset
            y1 = center_y - clone_h // 2 + y_offset
            x2 = x1 + clone_w
            y2 = y1 + clone_h
            
            # Clip clone to frame boundaries
            clone_x_start = max(0, -x1)
            clone_y_start = max(0, -y1)
            clone_x_end = clone_w - max(0, x2 - w)
            clone_y_end = clone_h - max(0, y2 - h)
            
            x1_clipped = max(0, x1)
            y1_clipped = max(0, y1)
            x2_clipped = min(w, x2)
            y2_clipped = min(h, y2)
            
            # Extract the valid portion of clone
            if clone_x_end > clone_x_start and clone_y_end > clone_y_start:
                clone_cropped = clone[clone_y_start:clone_y_end, clone_x_start:clone_x_end]
                
                # Get ROI from result frame
                roi = result_frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
                
                # Ensure dimensions match
                if roi.shape[:2] == clone_cropped.shape[:2]:
                    # Blend only an oval clone mask. Copying the entire frame
                    # creates nested room-sized rectangles instead of clones.
                    alpha = 0.5 + (i % 3) * 0.15  # Range from 0.5 to 0.8
                    mask = np.zeros((clone_cropped.shape[0], clone_cropped.shape[1]), dtype=np.uint8)
                    cv2.ellipse(
                        mask,
                        (clone_cropped.shape[1] // 2, clone_cropped.shape[0] // 2),
                        (max(1, clone_cropped.shape[1] // 3), max(1, clone_cropped.shape[0] // 2 - 4)),
                        0, 0, 360, int(255 * alpha), -1
                    )
                    mask = cv2.GaussianBlur(mask, (0, 0), 9)
                    mask = (mask.astype(np.float32) / 255.0)[..., None]
                    blended = roi.astype(np.float32) * (1.0 - mask) + clone_cropped.astype(np.float32) * mask
                    result_frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = blended.astype(np.uint8)
        
        # Add visual effect overlay (sparkles/particles) - yellow/golden effect
        result_frame = self.add_particle_effect(result_frame, color=(0, 200, 255), num_particles=80)
        
        # Add a subtle glow effect
        overlay = result_frame.copy()
        cv2.circle(overlay, (center_x, center_y), min(w, h) // 3, (0, 200, 255), -1)
        result_frame = cv2.addWeighted(result_frame, 0.85, overlay, 0.15, 0)
        
        return result_frame
    
    def fireball_jutsu(self, frame: np.ndarray) -> np.ndarray:
        """
        Add an animated fire effect
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()

        # Warm glow behind the flames.
        glow = np.zeros_like(result_frame)
        cv2.circle(glow, (w // 2, h // 2), min(w, h) // 5, (0, 70, 255), -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 45)
        result_frame = cv2.addWeighted(result_frame, 0.72, glow, 0.55, 0)

        # Draw layered flame tongues with small frame-to-frame variation.
        overlay = result_frame.copy()
        center_x, base_y = w // 2, int(h * 0.68)
        flame_colors = [(0, 40, 255), (0, 110, 255), (0, 190, 255), (80, 235, 255)]
        for i in range(18):
            x = center_x + np.random.randint(-min(w, h) // 7, min(w, h) // 7 + 1)
            width = np.random.randint(max(12, w // 45), max(18, w // 18))
            height = np.random.randint(max(35, h // 12), max(55, h // 5))
            y = base_y - np.random.randint(0, max(1, h // 10))
            points = np.array([
                [x - width, y], [x - width // 2, y - height // 2],
                [x - width // 4, y - height], [x, y - height // 2],
                [x + width // 3, y - height - np.random.randint(5, max(6, h // 18))],
                [x + width // 2, y - height // 3], [x + width, y]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [points], flame_colors[i % len(flame_colors)])

        result_frame = cv2.addWeighted(result_frame, 0.45, overlay, 0.75, 0)

        # Rising embers and smoke make the effect feel animated.
        result_frame = self.add_particle_effect(result_frame, color=(0, 170, 255), num_particles=65)
        smoke = result_frame.copy()
        for _ in range(8):
            x = center_x + np.random.randint(-w // 8, w // 8 + 1)
            y = np.random.randint(h // 5, h // 2)
            cv2.circle(smoke, (x, y), np.random.randint(8, 22), (80, 80, 80), -1)
        smoke = cv2.GaussianBlur(smoke, (0, 0), 18)
        result_frame = cv2.addWeighted(result_frame, 0.88, smoke, 0.12, 0)
        
        return result_frame

    def rasengan_jutsu(self, frame: np.ndarray) -> np.ndarray:
        """
        Add Rasengan effect
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        
        # Add blue swirling effect
        overlay = result_frame.copy()
        cv2.circle(overlay, (w // 2, h // 2), 150, (255, 200, 100), -1)
        result_frame = cv2.addWeighted(result_frame, 0.7, overlay, 0.3, 0)
        
        # Add particle effect
        result_frame = self.add_particle_effect(result_frame, color=(255, 200, 100))
        
        return result_frame
    
    def water_jutsu(self, frame: np.ndarray) -> np.ndarray:
        """
        Add water effect
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        
        # Add blue tint
        overlay = result_frame.copy()
        overlay[:, :] = (255, 200, 100)  # Blue tint
        result_frame = cv2.addWeighted(result_frame, 0.8, overlay, 0.2, 0)
        
        # Add particle effect
        result_frame = self.add_particle_effect(result_frame, color=(255, 200, 100))
        
        return result_frame
    
    def wind_jutsu(self, frame: np.ndarray) -> np.ndarray:
        """
        Add wind effect
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        
        # Add motion blur effect
        kernel = np.ones((5, 5), np.float32) / 25
        blurred = cv2.filter2D(result_frame, -1, kernel)
        result_frame = cv2.addWeighted(result_frame, 0.7, blurred, 0.3, 0)
        
        # Add particle effect
        result_frame = self.add_particle_effect(result_frame, color=(200, 200, 200))
        
        return result_frame
    
    def add_particle_effect(self, frame: np.ndarray, color: Tuple[int, int, int], 
                           num_particles: int = 50) -> np.ndarray:
        """
        Add particle/sparkle effect
        
        Args:
            frame: Input frame
            color: BGR color for particles
            num_particles: Number of particles
            
        Returns:
            frame: Frame with particles
        """
        h, w = frame.shape[:2]
        result_frame = frame.copy()
        
        # Generate random particle positions
        np.random.seed(int(time.time() * 1000) % 10000)
        for _ in range(num_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(2, 5)
            cv2.circle(result_frame, (x, y), size, color, -1)
        
        return result_frame
