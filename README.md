🎥 Real-Time Age & Gender Detection
📌 Overview

This project is a real-time Deep Learning-based system that predicts age and gender from live webcam input.

It integrates face detection with a trained Convolutional Neural Network (CNN) model to perform instant inference on detected faces.

The system is optimized for real-time performance using OpenCV and TensorFlow.

🧠 Problem Statement

Real-time demographic analysis has applications in:

Smart retail analytics

Human-computer interaction

Interactive systems

Surveillance and monitoring

This project demonstrates how Deep Learning models can be deployed in a live video pipeline for immediate inference.

🏗 System Pipeline

The system follows this real-time workflow:

Capture video from webcam

Detect faces using OpenCV

Preprocess detected face (resize, normalize)

Pass image to trained CNN model

Predict:

Gender (Male / Female)

Age (Regression output)

Display results on screen

🧬 Model Architecture

Convolutional Neural Network (CNN)

Multi-task learning:

Gender classification (Binary classification)

Age regression

Built using TensorFlow / Keras

⚙ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Pillow

🚀 Features

✔ Real-time webcam detection
✔ Multi-task prediction (Age + Gender)
✔ Face detection + Deep Learning inference
✔ Lightweight and fast

▶ Installation

Clone the repository:

git clone https://github.com/yourusername/age-gender-live-detection.git
cd age-gender-live-detection

Install dependencies:

pip install -r requirements.txt

Run the system:

python test.py
📊 Example Output

On-screen display:

Age: 24
Gender: Male

Predictions are shown in real-time above the detected face.

🔮 Future Improvements

Improve model accuracy with larger dataset

Optimize for GPU acceleration

Deploy as mobile application (TensorFlow Lite)

Convert into REST API service
