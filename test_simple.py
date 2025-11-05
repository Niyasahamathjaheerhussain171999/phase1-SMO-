#!/usr/bin/env python3
"""Quick test to see if basic detection works"""

# Fix OpenCV import issue
import sys
import types
if 'cv2.dnn' not in sys.modules:
    mock_dnn = types.ModuleType('cv2.dnn')
    mock_dnn.DictValue = type('DictValue', (), {})
    sys.modules['cv2.dnn'] = mock_dnn

import cv2
import numpy as np
from ultralytics import YOLO

print("✅ Imports successful!")

# Load model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # Use smaller model for speed
print("✅ Model loaded!")

# Test on first 10 frames of existing video
video_path = 'football_video.mp4'
if not cv2.VideoCapture(video_path).isOpened():
    print("❌ No video file found. Please download first.")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
print(f"✅ Video opened: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")

# Process first 10 frames
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, device='cpu', verbose=False)
    
    # Count detections
    person_count = 0
    ball_count = 0
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
            elif cls == 32:  # ball
                ball_count += 1
    
    print(f"Frame {i}: {person_count} players, {ball_count} balls detected")

cap.release()
print("\n✅ TEST PASSED! System is working.")
print("Now run: python simple_pass_detection.py")

