# Naruto Hand Sign Jutsu

A real-time hand gesture recognition project that triggers Naruto-inspired visual effects through your camera.

## Features

- 🤚 Real-time hand gesture detection using MediaPipe
- 🎭 Multiple jutsu gestures (Cloning, Fireball, Rasengan, and more)
- ✨ Visual effects triggered by gestures
- 📸 Cloning Jutsu: Creates ghost-like copies of yourself
- 🔥 Fireball flames, embers, and smoke effects
- 🌀 Rasengan visual effect support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the main application:
```bash
python main.py
```

### Train custom gestures:
```bash
python train_gesture.py
```

Choose option `3` to collect training data and train the model. For reliable results, collect at least 100 samples for each gesture in consistent lighting and with a clear background.

## Gestures

- **Cloning Jutsu**: Creates ghost-like clones
- **Fireball Jutsu**: Adds animated flames, embers, and smoke
- **Rasengan**: Adds a blue spinning energy effect

## Controls

- Press `Q` to quit
- Press `S` to save current frame
- Press `T` to toggle training mode

## Credits

- Original project and foundation: [SatyamSingh8449/NarutoJujutsu](https://github.com/SatyamSingh8449/NarutoJujutsu)
- Enhancements, Rasengan support, visual-effect improvements, compatibility fixes, and maintenance: **jajamchlramoji**

This is an independent derivative project. Please review the original repository's license and attribution requirements before redistributing it.
