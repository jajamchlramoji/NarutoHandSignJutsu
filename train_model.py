"""
Quick script to train the gesture model if training data exists
"""
from train_gesture import GestureTrainer

if __name__ == "__main__":
    trainer = GestureTrainer()
    print("Training model with existing data...")
    trainer.train(epochs=50)
