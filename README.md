# Fire Detection System using YOLOv8 & Computer Vision

This project detects fire in **images**, **videos**, and **real-time camera feeds** using a trained YOLOv8 model. It can also send real-time alerts `image and notification` via **Telegram** when fire is detected.

## Project Overview

Fire detection is critical for preventing loss of life and property.  
This project uses artificial intelligence and computer vision to automate fire recognition across different types of media:

- Static Images  
- Pre-recorded Videos  
- Real-time Camera Feeds

When fire is detected, the system highlights it and sends an immediate Telegram alert.

## Project Structure

```
Fire_Detection/
│
├── IMAGE_DONE.py              # Detect fire in a single image
├── VIDEO_DONEE.py             # Detect fire in a video file
├── Real_Time_DONE.py          # Detect fire from webcam or CCTV feed
├── best.pt                    # Trained YOLOv8 model weights
├── requirements.txt           # Required libraries
├── dataset.zip                # Dataset used for the model
└── README.md                  # Project documentation (this file)
```

## How Each Script Works

### IMAGE_DONE.py
- Loads an image from a specified path  
- Uses YOLOv8 to detect fire  
- If detected, draws bounding boxes and sends a Telegram alert

### VIDEO_DONEE.py
- Loads a video file  
- Processes it frame-by-frame  
- Detects fire, draws bounding boxes, and sends an alert if fire is found

### Real_Time_DONE.py
- Connects to a live camera stream (webcam or CCTV)  
- Detects fire in real time  
- Sends instant Telegram alerts with captured frames showing the fire

## Requirements

- Python 3.8+  
- Ultralytics (YOLOv8)  
- OpenCV  
- Requests

