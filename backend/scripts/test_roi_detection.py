import sys
import os
import cv2
import numpy as np

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing import detect_hands_in_image

def main():
    print("=== Testing ROI Hand Detection ===")
    
    # 1. Create a dummy BGR frame (320x240) containing a simulated "hand" (skin-colored square)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Fill background with a non-skin color (e.g., dark blue)
    frame[:] = [50, 20, 20]
    
    # Place a simulated hand (skin colored rectangle) in the center ROI
    # Center ROI is around x: 72..248, y: 36..204
    # Let's put a skin-colored box at x: 100..180, y: 80..160
    # Skin color in BGR: [100, 150, 200] (light brown/tan/pinkish)
    frame[80:160, 100:180] = [100, 150, 200]
    
    print("Running detection on frame with simulated hand inside ROI...")
    hands_detected, hand_count, bbox = detect_hands_in_image(frame)
    print(f"Result: detected={hands_detected}, count={hand_count}, bbox={bbox}")
    
    # 2. Run detection on an empty frame (no skin)
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    empty_frame[:] = [50, 20, 20]
    print("\nRunning detection on empty frame (no skin)...")
    hands_detected_empty, _, bbox_empty = detect_hands_in_image(empty_frame)
    print(f"Result: detected={hands_detected_empty}, bbox={bbox_empty}")

if __name__ == "__main__":
    main()
