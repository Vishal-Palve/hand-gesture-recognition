Hand Gesture Recognition using MediaPipe & Edge AI

This project implements a real-time hand-gesture recognition system using MediaPipe, OpenCV, and a lightweight MLP/TFLite model.
It runs entirely on the local device (Edge AI) without needing any cloud processing.

The application uses webcam input to:

Detect a single hand

Extract 21 key landmark points

Classify the gesture into predefined classes

Display the gesture result live on screen

This project is useful for:

Touchless interfaces

Human–computer interaction

Controlling systems using gestures

Learning Edge AI / MediaPipe pipelines

🎥 Demo (Run on Webcam)

Run the main script:

python app.py

Optional arguments:
Option	Description	Default
--device	Camera device index	0
--width	Webcam capture width	960
--height	Webcam capture height	540
--use_static_image_mode	Enable MediaPipe static mode	Off
--min_detection_confidence	Confidence threshold	0.5
--min_tracking_confidence	Tracking threshold	0.5
📁 Project Structure
│  app.py                          # Real-time gesture pipeline
│  keypoint_classification.ipynb   # Training notebook for gesture classifier
│  point_history_classification.ipynb # Training for motion-based gestures
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv                     # Training dataset
│  │  │  keypoint_classifier.tflite       # Trained TFLite model
│  │  │  keypoint_classifier.hdf5         # Keras model (optional)
│  │  │  keypoint_classifier.py           # Inference module
│  │  └─ keypoint_classifier_label.csv    # Label mapping
│  │
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.tflite
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      └─ point_history_classifier_label.csv
│
└─utils
    └─cvfpscalc.py          # FPS counter utility

🧠 How It Works
1. Landmark Detection (MediaPipe Hands)

MediaPipe detects:

21 hand landmarks

Normalizes and preprocesses coordinates

Sends them to the classifier

2. Gesture Classification

Two classifiers are supported:

A. Hand Sign Recognition (Static Gestures)

Examples:

Open hand

Closed fist

Pointing

Thumbs up

Victory (✌️), etc.

B. Finger Movement Recognition (Dynamic Gestures)

Uses fingertip coordinate history to detect:

Circular motions

Swiping

Movement direction

Other motion-based gestures

🎓 Training the Model
Static Hand Gesture Model

Run the app

Press k → activates keypoint recording mode

Press 0–9 → saves labeled keypoint data into keypoint.csv

Open keypoint_classification.ipynb

Run all cells to train a new model

Export updated TFLite model

Dynamic Gesture Model

Press h → logs index finger movement

Press 0–9 → saves motion sequences

Open point_history_classification.ipynb

Retrain

Export new model

⚡ Model Highlights
Static Gesture Model

Input: 21 × (x,y,z) landmarks

Architecture: Simple MLP

Optimized for edge devices

Runs in real time (~30–90 FPS depending on system)

Dynamic Gesture Model

Input: Time-series of fingertip coordinates

Uses MLP or LSTM (Optional)

Supports custom motions

Light enough for CPU-only inference

🚀 Technologies Used

MediaPipe Hands

OpenCV

Python

TensorFlow / TFLite

NumPy

Matplotlib / scikit-learn (for optional visualization)

🛠 Requirements
mediapipe >= 0.8
opencv-python >= 4.0
tensorflow >= 2.3
scikit-learn >= 0.23   # Optional
matplotlib >= 3.3       # Optional


Install via:

pip install -r requirements.txt


(or install manually)

🧩 Features

✔ Runs fully offline (Edge AI)

✔ Supports custom gestures (static + dynamic)

✔ Real-time FPS

✔ Re-trainable with your own dataset

✔ Lightweight (TFLite model < 50 KB)

📘 How to Extend

You can add:

Air drawing

Gesture-controlled volume

Virtual mouse

Home automation gesture control

Sign language recognition

I can help you build any of these too.

📜 License

This project is released under your own license:
© 2025 Vishal Palve – All rights reserved.
(Modify based on how open-source you want it to be)


Professional

Perfect for GitHub + Resume
