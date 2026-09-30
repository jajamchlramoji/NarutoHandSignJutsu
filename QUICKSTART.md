# Quick Start Guide

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## First Time Setup

### Step 1: Train the Model

Before you can use the jutsu recognition, you need to train the model with your hand gestures:

```bash
python train_gesture.py
```

Choose option `1` to collect training data:
- Press number keys `0-4` to select which gesture you're performing
- Hold the gesture in front of the camera
- Collect at least 50 samples per gesture (more is better!)
- Press `Q` when done collecting

Then choose option `2` to train the model, or option `3` to do both.

### Step 2: Run the Application

```bash
python main.py
```

## How to Use

1. **Perform Hand Gestures**: Make hand gestures in front of your camera
2. **Watch for Effects**: When a gesture is recognized with high confidence (>70%), the corresponding jutsu effect will appear
3. **Cloning Jutsu**: When detected, creates 10 clones of yourself arranged in a circle
4. **Controls**:
   - `Q` - Quit application
   - `S` - Save screenshot

## Gesture Classes

- **0**: None (background/no gesture)
- **1**: Cloning Jutsu (creates 10 clones)
- **2**: Fireball Jutsu (fire effect)
- **3**: Water Jutsu (water effect)
- **4**: Wind Jutsu (wind effect)

## Tips for Best Results

1. **Good Lighting**: Ensure your hands are well-lit
2. **Clear Background**: Use a plain background for better hand detection
3. **Consistent Gestures**: Try to perform gestures consistently during training
4. **More Training Data**: Collect more samples (100+ per gesture) for better accuracy
5. **Hold Gestures**: Hold gestures for a moment to trigger effects

## Troubleshooting

- **Camera not opening**: Check if another application is using the camera
- **Low accuracy**: Collect more training data and retrain
- **Effects not appearing**: Make sure confidence threshold is met (>70%)
- **Model not found**: Run `train_gesture.py` first to create the model
