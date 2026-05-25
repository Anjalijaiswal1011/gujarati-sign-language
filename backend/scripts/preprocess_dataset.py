import os
import cv2
import csv
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed_data")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "landmarks.csv")

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Get only the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Normalize relative to wrist (landmark 0)
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        
        normalized_landmarks = []
        for lm in hand_landmarks.landmark:
            normalized_landmarks.append(lm.x - base_x)
            normalized_landmarks.append(lm.y - base_y)
        return normalized_landmarks
    return None

def main():
    print("=== Extracting Landmarks ===")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Directory {RAW_IMAGES_DIR} not found.")
        return
        
    labels = [d for d in os.listdir(RAW_IMAGES_DIR) if os.path.isdir(os.path.join(RAW_IMAGES_DIR, d))]
    if not labels:
        print("No label directories found in raw_images/")
        return
        
    print(f"Found labels: {labels}")
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: class_id, x0, y0, ..., x20, y20
        header = ['class_id', 'class_name'] + [f'v_{i}' for i in range(42)]
        writer.writerow(header)
        
        for class_id, label in enumerate(labels):
            label_dir = os.path.join(RAW_IMAGES_DIR, label)
            images = [img for img in os.listdir(label_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
            
            # Limit to 500 images per class to keep training time reasonable (under 10 mins)
            MAX_IMAGES_PER_CLASS = 500
            images = images[:MAX_IMAGES_PER_CLASS]
            
            print(f"Processing {label}: {len(images)} images")
            for img_name in images:
                img_path = os.path.join(label_dir, img_name)
                landmarks = process_image(img_path)
                if landmarks:
                    row = [class_id, label] + landmarks
                    writer.writerow(row)

    print(f"Preprocessing complete. Data saved to {OUTPUT_CSV}")
    
    # Save a temporary label mapping in processed dir
    mapping_file = os.path.join(PROCESSED_DIR, "class_mapping.txt")
    with open(mapping_file, "w") as f:
        for class_id, label in enumerate(labels):
            f.write(f"{class_id},{label}\n")

if __name__ == "__main__":
    main()
