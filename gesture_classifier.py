"""
CNN-based gesture classifier for recognizing Naruto jutsu gestures
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

class GestureClassifier(nn.Module):
    """
    Simple CNN classifier for hand gesture recognition
    Input: 63 features (21 landmarks * 3 coordinates)
    """
    def __init__(self, num_classes=6, input_size=63):
        super(GestureClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class JutsuRecognizer:
    """
    Wrapper class for gesture recognition
    """
    def __init__(self, model_path='models/gesture_model.pth', num_classes=6):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the checkpoint first so models trained with an older number of
        # gesture classes remain compatible with the current application.
        checkpoint = None
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            output_weights = checkpoint.get('fc4.weight')
            if output_weights is not None:
                num_classes = output_weights.shape[0]

        # Initialize model with the same output size as the saved checkpoint.
        self.model = GestureClassifier(num_classes=num_classes)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}. Using untrained model.")
            print("Please train the model first using train_gesture.py")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Jutsu names
        self.jutsu_names = {
            0: "None",
            1: "Cloning Jutsu",
            2: "Fireball Jutsu",
            3: "Water Jutsu",
            4: "Wind Jutsu",
            5: "Rasengan"
        }
        
    def predict(self, landmarks):
        """
        Predict gesture from landmarks
        
        Args:
            landmarks: Array of shape (63,) or list of arrays
            
        Returns:
            prediction: Jutsu name string
            confidence: Confidence score
        """
        if landmarks is None or len(landmarks) == 0:
            return "None", 0.0
        
        # Use first hand if multiple hands detected
        if isinstance(landmarks, list):
            landmarks = landmarks[0]
        
        # Convert to tensor
        landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        # Only return prediction if confidence is high enough
        if confidence_score > 0.7:
            return self.jutsu_names.get(predicted_class, "None"), confidence_score
        else:
            return "None", confidence_score
