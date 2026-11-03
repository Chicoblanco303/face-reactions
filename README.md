# Face Reactions – Python MediaPipe Project

A real-time facial expression and hand gesture recognition application built using Python, OpenCV, and MediaPipe.

This project uses a webcam to analyze facial landmarks and hand gestures, then displays different reaction images depending on the detected expressions and gestures.

## Project Preview

Check the assets (screenshot sample)

## Features

- Real-time facial landmark detection.
- Hand tracking and gesture recognition.
- Facial expression analysis.
- Heart hand gesture detection.
- Hand-near-ear gesture detection.
- Dynamic reaction image display.
- Live debugging information.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Face Mesh and hand tracking |
| NumPy | Image array processing |

## Reaction Detection

| Detected Expression or Gesture | Reaction |
|---|---|
| Neutral face | Ellie |
| Smiling | Smiling Apple |
| Open mouth | Hamster / Wojak |
| Closed eyes | IShowSpeed |
| Frowning | Sad Emoji |
| Hand near ear | Kanye |
| Heart hand gesture | Kratos |
| Slightly open mouth | Apple |

## Project Structure

```text
face-reactions/
│
├── face_reactions.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── assets/
    ├── images/
    │   └── Reaction images
    │
    └── screenshots/
        └── face_reactions_demo_no_video.png
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Chicoblanco303/face-reactions.git
```

2. Navigate to the project directory:

```bash
cd face-reactions
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

1. Make sure your webcam is connected.
2. Run `face_reactions.py` using Python IDLE or a terminal.
3. Allow webcam access if prompted.
4. Perform different facial expressions and hand gestures.
5. Press **Q** to close the application.

## Project Purpose

This project was developed to explore computer vision, facial landmark detection, hand gesture recognition, and real-time image processing using Python.

## Developer

**Shintaro Braza**

BSIT Student | Python | IT Support | Graphic Design

GitHub: [Chicoblanco303](https://github.com/Chicoblanco303)
