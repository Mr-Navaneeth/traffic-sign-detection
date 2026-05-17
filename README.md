# Traffic Sign Detection System using YOLOv8

## Overview
This project is a real-time Traffic Sign Detection System developed using the YOLOv8 object detection model.  
The system can detect and classify multiple traffic signs from images, videos, and webcam input.

The project was trained using a custom dataset containing 44 traffic sign classes.

## Features
- Real-time traffic sign detection
- Image, video, and webcam support
- Bounding box visualization with labels
- YOLOv8 custom trained model
- Fast and efficient detection

## 🧠 Model Used
- YOLOv8 (Ultralytics)
- Python
- OpenCV

## 📂 Dataset Details
- Total Images: 1079
- Total Classes: 44
- Annotation Format: YOLO format

> Some classes contain fewer training images, which may affect accuracy.

## ⚙️ Installation

```bash
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection

pip install -r requirements.txt
