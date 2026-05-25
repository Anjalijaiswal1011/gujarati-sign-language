"""
Landmark Extraction Service
============================
Extracts MediaPipe hand landmarks from images and videos,
normalizes them, and stores as sequences for model training.

Usage:
    from services.landmark_extraction import extract_landmarks_from_image, extract_landmarks_from_video
"""

import os
import numpy as np
import cv2
from pathlib import Path

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "services", "models")
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# Initialize MediaPipe
_mp_hands_detector = None

def initialize_landmarks_detector():
    """Initialize MediaPipe hand landmark detector."""
    global _mp_hands_detector
    
    if _mp_hands_detector is not None:
        return _mp_hands_detector
    
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print(f"[Landmark] Model not found at {LANDMARK_MODEL_PATH}")
        return None
    
    try:
        base_options = python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,  # Detect up to 2 hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        _mp_hands_detector = vision.HandLandmarker.create_from_options(options)
        print("[Landmark] MediaPipe HandLandmarker initialized successfully")
        return _mp_hands_detector
    except Exception as e:
        print(f"[Landmark] Failed to initialize MediaPipe: {e}")
        return None


def normalize_landmarks(landmarks, image_width, image_height):
    """
    Normalize landmarks to [-1, 1] range based on hand bounding box.
    
    Args:
        landmarks: numpy array of shape (21, 4) - 21 points with (x, y, z, confidence)
        image_width: width of original image
        image_height: height of original image
    
    Returns:
        normalized_landmarks: numpy array of shape (84,) - flattened normalized landmarks
    """
    if landmarks is None or len(landmarks) == 0:
        return None
    
    landmarks = np.array(landmarks)
    
    # Get bounding box
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    z_coords = landmarks[:, 2]
    
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    # Hand center and size
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    hand_size = max(x_max - x_min, y_max - y_min, 0.01)  # Avoid division by zero
    
    # Normalize
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, 0] = (landmarks[:, 0] - x_center) / (hand_size / 2)
    landmarks_normalized[:, 1] = (landmarks[:, 1] - y_center) / (hand_size / 2)
    landmarks_normalized[:, 2] = landmarks[:, 2] / (hand_size / 2)  # Z coordinate
    # Confidence remains as is
    
    return landmarks_normalized.flatten()


def extract_landmarks_from_image(image_path):
    """
    Extract hand landmarks from a single image file.
    
    Args:
        image_path: path to image file
    
    Returns:
        landmarks: normalized landmarks (84,) or None if no hands detected
    """
    detector = initialize_landmarks_detector()
    if detector is None:
        return None
    
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[Landmark] Failed to read image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=image_rgb)
        
        # Detect landmarks
        result = detector.detect(mp_image)
        
        if not result.hand_landmarks:
            return None
        
        # Get first hand landmarks
        hand_landmarks = result.hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in hand_landmarks]
        
        # Normalize
        h, w = image.shape[:2]
        normalized = normalize_landmarks(landmarks, w, h)
        
        return normalized
    
    except Exception as e:
        print(f"[Landmark] Error extracting landmarks from {image_path}: {e}")
        return None


def extract_landmarks_from_video(video_path, frame_skip=1, max_frames=None):
    """
    Extract hand landmarks from video frames.
    
    Args:
        video_path: path to video file
        frame_skip: process every Nth frame (1 = all frames, 2 = every 2nd frame)
        max_frames: maximum number of frames to extract (None = all)
    
    Returns:
        landmarks_sequence: numpy array of shape (num_frames, 84) or None
    """
    detector = initialize_landmarks_detector()
    if detector is None:
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Landmark] Failed to open video: {video_path}")
            return None
        
        frame_count = 0
        all_landmarks = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Stop at max_frames
            if max_frames and len(all_landmarks) >= max_frames:
                break
            
            # Extract landmarks
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=image_rgb)
                result = detector.detect(mp_image)
                
                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in hand_landmarks]
                    h, w = frame.shape[:2]
                    normalized = normalize_landmarks(landmarks, w, h)
                    
                    if normalized is not None:
                        all_landmarks.append(normalized)
            
            except Exception as e:
                print(f"[Landmark] Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        if not all_landmarks:
            print(f"[Landmark] No hand landmarks found in video: {video_path}")
            return None
        
        return np.array(all_landmarks)
    
    except Exception as e:
        print(f"[Landmark] Error extracting landmarks from video {video_path}: {e}")
        return None


def pad_or_truncate_sequence(sequence, target_length=30):
    """
    Pad sequence to target length or truncate if longer.
    
    Args:
        sequence: numpy array of shape (num_frames, 84)
        target_length: desired sequence length
    
    Returns:
        padded_sequence: numpy array of shape (target_length, 84)
    """
    if sequence is None or len(sequence) == 0:
        return None
    
    current_length = len(sequence)
    feature_dim = sequence.shape[1] if len(sequence.shape) > 1 else 84
    
    if current_length == target_length:
        return sequence
    
    if current_length > target_length:
        # Truncate from middle if too long
        start_idx = (current_length - target_length) // 2
        return sequence[start_idx:start_idx + target_length]
    
    # Pad with zeros if too short
    padded = np.zeros((target_length, feature_dim))
    padded[:current_length] = sequence
    return padded


def augment_sequence(sequence, factor=3, seed=42):
    """
    Augment a landmark sequence to increase dataset size.
    
    Args:
        sequence: numpy array of shape (num_frames, 84)
        factor: number of augmentations to create
        seed: random seed for reproducibility
    
    Returns:
        augmented_sequences: list of numpy arrays
    """
    if sequence is None:
        return None
    
    np.random.seed(seed)
    augmented = [sequence]
    
    for i in range(factor - 1):
        aug_seq = sequence.copy()
        
        # 1. Scaling (±15% size variation)
        scale = np.random.uniform(0.85, 1.15, (len(aug_seq), 1))
        aug_seq *= scale
        
        # 2. Noise (small random perturbation)
        noise = np.random.normal(0, 0.012, aug_seq.shape)
        aug_seq += noise
        
        # 3. Translation jitter (same for all frames in sequence)
        jitter = np.random.normal(0, 0.025, (1, aug_seq.shape[1]))
        aug_seq += jitter
        
        # 4. Temporal warping (stretch/compress sequence)
        if i % 2 == 0:
            indices = np.linspace(0, len(aug_seq) - 1, len(aug_seq)).astype(int)
            indices = np.clip(indices + np.random.randint(-1, 2, len(indices)), 0, len(aug_seq) - 1)
            aug_seq = aug_seq[indices]
        
        augmented.append(aug_seq)
    
    return augmented


def process_word_dataset(dataset_root, output_root, target_length=30, augment=True):
    """
    Process word gesture images into landmark sequences.
    
    Args:
        dataset_root: root directory with word subdirectories
        output_root: output directory for sequences
        target_length: target sequence length
        augment: whether to apply data augmentation
    
    Returns:
        summary: dict with statistics
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_words": 0,
        "total_sequences": 0,
        "total_errors": 0,
        "word_stats": {}
    }
    
    # Process each word directory
    for word_dir in sorted(dataset_path.iterdir()):
        if not word_dir.is_dir():
            continue
        
        word_name = word_dir.name
        word_output = output_path / word_name
        word_output.mkdir(exist_ok=True)
        
        image_files = list(word_dir.glob("*.jpg")) + list(word_dir.glob("*.png"))
        
        summary["total_words"] += 1
        summary["word_stats"][word_name] = {
            "images": len(image_files),
            "sequences": 0,
            "errors": 0
        }
        
        print(f"\nProcessing word: {word_name} ({len(image_files)} images)")
        
        seq_count = 0
        for idx, img_file in enumerate(image_files):
            landmarks = extract_landmarks_from_image(str(img_file))
            
            if landmarks is None:
                summary["total_errors"] += 1
                summary["word_stats"][word_name]["errors"] += 1
                continue
            
            # Each image becomes a sequence (single frame repeated)
            sequence = np.repeat([landmarks], target_length, axis=0)
            
            # Save original
            seq_name = f"{word_name}_{idx:05d}.npy"
            np.save(str(word_output / seq_name), sequence)
            seq_count += 1
            
            # Augment if requested
            if augment:
                augmented = augment_sequence(sequence, factor=3)
                for aug_idx, aug_seq in enumerate(augmented[1:], 1):
                    aug_name = f"{word_name}_{idx:05d}_aug{aug_idx}.npy"
                    np.save(str(word_output / aug_name), aug_seq)
                    seq_count += 1
        
        summary["total_sequences"] += seq_count
        summary["word_stats"][word_name]["sequences"] = seq_count
        print(f"  → Saved {seq_count} sequences")
    
    return summary


def process_action_dataset(dataset_root, output_root, target_length=30, frame_skip=1, augment=True):
    """
    Process action videos into landmark sequences.
    
    Args:
        dataset_root: root directory with action subdirectories
        output_root: output directory for sequences
        target_length: target sequence length
        frame_skip: process every Nth frame
        augment: whether to apply data augmentation
    
    Returns:
        summary: dict with statistics
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_actions": 0,
        "total_sequences": 0,
        "total_errors": 0,
        "action_stats": {}
    }
    
    # Process each action directory
    for action_dir in sorted(dataset_path.iterdir()):
        if not action_dir.is_dir():
            continue
        
        action_name = action_dir.name
        action_output = output_path / action_name
        action_output.mkdir(exist_ok=True)
        
        video_files = list(action_dir.glob("*.mp4")) + list(action_dir.glob("*.avi")) + list(action_dir.glob("*.mov"))
        
        summary["total_actions"] += 1
        summary["action_stats"][action_name] = {
            "videos": len(video_files),
            "sequences": 0,
            "errors": 0
        }
        
        print(f"\nProcessing action: {action_name} ({len(video_files)} videos)")
        
        seq_count = 0
        for idx, video_file in enumerate(video_files):
            landmarks = extract_landmarks_from_video(str(video_file), frame_skip=frame_skip, max_frames=target_length * 2)
            
            if landmarks is None:
                summary["total_errors"] += 1
                summary["action_stats"][action_name]["errors"] += 1
                continue
            
            # Pad/truncate to target length
            sequence = pad_or_truncate_sequence(landmarks, target_length)
            
            if sequence is None:
                summary["total_errors"] += 1
                summary["action_stats"][action_name]["errors"] += 1
                continue
            
            # Save original
            seq_name = f"{action_name}_{idx:05d}.npy"
            np.save(str(action_output / seq_name), sequence)
            seq_count += 1
            
            # Augment if requested
            if augment:
                augmented = augment_sequence(sequence, factor=3)
                for aug_idx, aug_seq in enumerate(augmented[1:], 1):
                    aug_name = f"{action_name}_{idx:05d}_aug{aug_idx}.npy"
                    np.save(str(action_output / aug_name), aug_seq)
                    seq_count += 1
        
        summary["total_sequences"] += seq_count
        summary["action_stats"][action_name]["sequences"] = seq_count
        print(f"  → Saved {seq_count} sequences")
    
    return summary


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract landmarks from gestures")
    parser.add_argument("--mode", choices=["words", "actions", "both"], default="both")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-root", default="dataset/sequences")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    base_dir = BASE_DIR
    dataset_root = os.path.join(base_dir, args.dataset_root)
    output_root = os.path.join(base_dir, args.output_root)
    
    all_summary = {}
    
    if args.mode in ["words", "both"]:
        print("=" * 60)
        print("EXTRACTING LANDMARKS FROM WORD GESTURES")
        print("=" * 60)
        words_root = os.path.join(dataset_root, "words")
        words_output = os.path.join(output_root, "words")
        
        if os.path.exists(words_root):
            summary = process_word_dataset(words_root, words_output, augment=not args.no_augment)
            all_summary["words"] = summary
            print("\n" + "=" * 60)
            print("WORD PROCESSING SUMMARY")
            print("=" * 60)
            print(json.dumps(summary, indent=2))
        else:
            print(f"Word dataset directory not found: {words_root}")
    
    if args.mode in ["actions", "both"]:
        print("\n" + "=" * 60)
        print("EXTRACTING LANDMARKS FROM ACTION GESTURES")
        print("=" * 60)
        actions_root = os.path.join(dataset_root, "actions")
        actions_output = os.path.join(output_root, "actions")
        
        if os.path.exists(actions_root):
            summary = process_action_dataset(actions_root, actions_output, augment=not args.no_augment)
            all_summary["actions"] = summary
            print("\n" + "=" * 60)
            print("ACTION PROCESSING SUMMARY")
            print("=" * 60)
            print(json.dumps(summary, indent=2))
        else:
            print(f"Action dataset directory not found: {actions_root}")
    
    # Save summary
    summary_path = os.path.join(output_root, "extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
