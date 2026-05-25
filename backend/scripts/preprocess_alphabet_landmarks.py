import os
import sys
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

# Add backend directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from services.preprocessing import extract_landmarks
from services.constants import ALPHABET_LABELS

RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed_data")
CACHE_FILE = os.path.join(PROCESSED_DIR, "alphabet_landmarks_cache.npz")

def extract_single_image(args):
    img_path, class_idx = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        lm = extract_landmarks(img)
        if lm is not None:
            return lm, class_idx
    except Exception:
        pass
    return None

def main():
    print("=== PRE-EXTRACTING ALPHABET LANDMARKS (PARALLEL) ===")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"Directory {RAW_IMAGES_DIR} not found.")
        return
        
    all_dirs = [d for d in os.listdir(RAW_IMAGES_DIR)
                if os.path.isdir(os.path.join(RAW_IMAGES_DIR, d))]
    alphabet_lower = [l.lower() for l in ALPHABET_LABELS]
    target_dirs = sorted([d for d in all_dirs if d.lower() in alphabet_lower])
    
    class_to_idx = {name: idx for idx, name in enumerate(target_dirs)}
    
    jobs = []
    # Use up to 800 images per class for balanced accuracy and speed
    MAX_IMAGES = 800 
    
    for name in target_dirs:
        class_dir = os.path.join(RAW_IMAGES_DIR, name)
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(files)
        for f in files[:MAX_IMAGES]:
            jobs.append((os.path.join(class_dir, f), class_to_idx[name]))
            
    print(f"Total jobs scheduled: {len(jobs)} across {len(target_dirs)} classes.")
    print(f"Processing in parallel using {cpu_count()} CPU cores...")
    
    X_data = []
    y_data = []
    
    # Run multiprocessing pool
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_single_image, jobs)
        
    for res in results:
        if res is not None:
            lm, idx = res
            X_data.append(lm)
            y_data.append(idx)
            
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    
    print(f"Extraction finished. Valid hands found: {len(X)} / {len(jobs)}")
    
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=np.array(target_dirs))
    print(f"Cached landmarks saved to {CACHE_FILE}")

if __name__ == "__main__":
    main()
