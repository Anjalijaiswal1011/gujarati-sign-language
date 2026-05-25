import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from services.gesture_service import process_gesture_frame
import cv2
import base64

def main():
    print("Testing Gesture Service...")
    # Create a dummy blank image
    import numpy as np
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add a "hand" (a skin colored blob)
    cv2.circle(img, (320, 240), 100, (180, 200, 240), -1) 
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    
    res = process_gesture_frame(b64, mode='words')
    print("Result:", res)

if __name__ == "__main__":
    main()
