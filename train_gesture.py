"""
Training script for gesture recognition model
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import pickle
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier

class GestureDataset(Dataset):
    """Dataset for hand gesture training"""
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.landmarks = []
        self.labels = []
        self.load_data()
        
    def load_data(self):
        """Load training data from files"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
            print("Please collect training data first!")
            return
        
        # Load saved landmarks and labels
        landmarks_file = os.path.join(self.data_dir, 'landmarks.pkl')
        labels_file = os.path.join(self.data_dir, 'labels.pkl')
        
        if os.path.exists(landmarks_file) and os.path.exists(labels_file):
            with open(landmarks_file, 'rb') as f:
                self.landmarks = pickle.load(f)
            with open(labels_file, 'rb') as f:
                self.labels = pickle.load(f)
            print(f"Loaded {len(self.landmarks)} samples")
        else:
            print("No training data found. Please collect data first!")
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.landmarks[idx]), torch.LongTensor([self.labels[idx]])[0]

class GestureTrainer:
    """Trainer for gesture recognition"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hand_detector = HandDetector()
        self.jutsu_classes = {
            0: "None",
            1: "Cloning Jutsu",
            2: "Fireball Jutsu",
            3: "Water Jutsu",
            4: "Wind Jutsu",
            5: "Rasengan"
        }
        
    def collect_data(self, num_samples_per_class=50):
        """Collect training data from camera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera")
            return
        
        print("\n=== Data Collection Mode ===")
        print("Press number keys to record gestures:")
        print("  0: None (background)")
        print("  1: Cloning Jutsu")
        print("  2: Fireball Jutsu")
        print("  3: Water Jutsu")
        print("  4: Wind Jutsu")
        print("Press 'Q' to finish collection")
        
        landmarks_list = []
        labels_list = []
        current_label = None
        samples_collected = {i: 0 for i in range(6)}
        
        os.makedirs('data', exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            results = self.hand_detector.detect_hands(frame)
            frame = self.hand_detector.draw_landmarks(frame, results)
            
            # Display current status
            status_text = f"Current Gesture: {self.jutsu_classes.get(current_label, 'None')}"
            cv2.putText(frame, status_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for i, (label, name) in enumerate(self.jutsu_classes.items()):
                count = samples_collected[label]
                cv2.putText(frame, f"{label}: {name} ({count}/{num_samples_per_class})", 
                           (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('5'):
                current_label = key - ord('0')
                print(f"Selected gesture: {self.jutsu_classes[current_label]}")
            
            # Collect sample if label is selected and hand is detected
            if current_label is not None:
                landmarks = self.hand_detector.extract_landmarks(results, frame.shape[:2])
                if landmarks:
                    landmarks_list.append(landmarks[0])
                    labels_list.append(current_label)
                    samples_collected[current_label] += 1
                    print(f"Collected sample {samples_collected[current_label]} for {self.jutsu_classes[current_label]}")
                    
                    # Reset if enough samples collected
                    if samples_collected[current_label] >= num_samples_per_class:
                        print(f"Enough samples collected for {self.jutsu_classes[current_label]}")
                        current_label = None
            
            time.sleep(0.1)  # Small delay to avoid collecting too fast
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save collected data
        if landmarks_list:
            with open('data/landmarks.pkl', 'wb') as f:
                pickle.dump(landmarks_list, f)
            with open('data/labels.pkl', 'wb') as f:
                pickle.dump(labels_list, f)
            print(f"\nSaved {len(landmarks_list)} samples to data/")
        else:
            print("No data collected!")
    
    def train(self, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the gesture classifier"""
        # Load dataset
        dataset = GestureDataset()
        if len(dataset) == 0:
            print("No training data available. Please collect data first!")
            return
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = GestureClassifier(num_classes=6).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"\n=== Training Model ===")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for landmarks, labels in dataloader:
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(landmarks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/gesture_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    import time
    
    trainer = GestureTrainer()
    
    print("=== Naruto Jutsu Gesture Trainer ===")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Both (collect then train)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        num_samples = int(input("Samples per class (default 50): ") or "50")
        trainer.collect_data(num_samples_per_class=num_samples)
    elif choice == "2":
        epochs = int(input("Epochs (default 50): ") or "50")
        trainer.train(epochs=epochs)
    elif choice == "3":
        num_samples = int(input("Samples per class (default 50): ") or "50")
        trainer.collect_data(num_samples_per_class=num_samples)
        epochs = int(input("Epochs (default 50): ") or "50")
        trainer.train(epochs=epochs)
    else:
        print("Invalid choice!")
