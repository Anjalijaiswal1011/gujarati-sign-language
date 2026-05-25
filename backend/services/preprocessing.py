"""
Image Preprocessing Pipeline
Ported from: github.com/mrrayyan2005/learning-app-for-mute-and-deaf (app.py)

Uses OpenCV-based hand detection (skin color + contour analysis)
instead of MediaPipe, matching the reference project's approach.
"""
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import base64
import io
from PIL import Image

try:
    import os
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Path to hand landmarker model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "services", "models", "hand_landmarker.task")
    
    if os.path.exists(model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55
        )
        _mp_hands_detector = vision.HandLandmarker.create_from_options(options)
        print("[Preprocessing] MediaPipe Tasks HandLandmarker initialized successfully.")
    else:
        _mp_hands_detector = None
        print(f"[Preprocessing] MediaPipe task model not found at {model_path}")
except Exception as e:
    mp = None
    _mp_hands_detector = None
    print(f"[Preprocessing] MediaPipe tasks initialization failed: {e}")




# ──────────────────────────────────────────────
# Constants (matching the reference Flask app)
# ──────────────────────────────────────────────
MODEL_INPUT_SIZE = 300   # Both classifiers expect 300×300 images


def decode_base64_image(base64_str):
    """
    Decodes a base64 data-URI or raw base64 string into an OpenCV BGR image.
    Returns None on any error.
    """
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        print(f"[Preprocessing] Decode error: {e}")
        return None


def detect_hands_in_image(cv_image):
    """
    Detects whether a hand is visible in the frame using:
      • Google MediaPipe Hands (robust landmark-based tracking if available)
      • Fallback: Center-ROI skin-colour contour analysis (to prevent face/background triggers)

    Returns (hands_detected: bool, hand_count: int, bbox: tuple)
    """
    global _mp_hands_detector
    
    # Black out faces to prevent face skin from triggering false skin detection
    # and to prevent face features from being classified as hand signs by the CNN.
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not face_cascade.empty():
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(cv_image, (fx, fy), (fx+fw, fy+fh), (0, 0, 0), -1)
    except Exception as fe:
        print(f"[Hand Detection] Face masking error: {fe}")
    
    # ── 1. Try MediaPipe Hands Detection (Primary) ──────────────────────────
    if _mp_hands_detector is not None:
        try:
            h_img, w_img, _ = cv_image.shape
            # Convert BGR to RGB
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image_rgb)
            results = _mp_hands_detector.detect(mp_image)
            
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                
                # Extract landmark coordinates
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                
                # Convert normalized coordinates to pixels
                x_min = int(min(x_coords) * w_img)
                x_max = int(max(x_coords) * w_img)
                y_min = int(min(y_coords) * h_img)
                y_max = int(max(y_coords) * h_img)
                
                # Add padding around the hand box
                padding = 30
                x = max(0, x_min - padding)
                y = max(0, y_min - padding)
                w = min(w_img - x, (x_max - x_min) + 2 * padding)
                h = min(h_img - y, (y_max - y_min) + 2 * padding)
                
                bbox = (x, y, w, h)
                print(f"[Hand Detection] MediaPipe SUCCESS: bbox={bbox}")
                return True, len(results.hand_landmarks), bbox
                
            # If MediaPipe processed but found absolutely no hand, do not predict anything.
            # This prevents false predictions (like predicting letters on faces/backgrounds).
            return False, 0, None
            
        except Exception as e:
            print(f"[Hand Detection] MediaPipe process error: {e}. Falling back to Center-ROI...")
            
    # ── 2. Fallback: Center-ROI skin color ratio tracker ───────────────────────
    # Since MediaPipe solutions is missing on Windows Python 3.13, this is the main robust path.
    try:
        h_img, w_img, _ = cv_image.shape
        
        # Define center ROI bounding box (matching frontend guide box)
        w_box = int(w_img * 0.55) # 55% of frame width
        h_box = int(h_img * 0.70) # 70% of frame height
        x_start = (w_img - w_box) // 2
        y_start = (h_img - h_box) // 2
        
        # Crop to the Region of Interest (ROI)
        roi_img = cv_image[y_start : y_start + h_box, x_start : x_start + w_box]
        
        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Robust skin color boundaries: Hue (0-25), Saturation (15-255), Value (30-255)
        # Using more lenient boundaries to support various lighting conditions and skin tones.
        lower_skin = np.array([0, 15, 30], dtype=np.uint8) 
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = np.sum(skin_mask > 0)
        total_roi_pixels = w_box * h_box
        skin_ratio = skin_pixels / total_roi_pixels
        
        # Trigger hand detection contour analysis if skin ratio is above 2%
        if skin_ratio > 0.02:
            # Find coordinates of skin pixels inside ROI
            coords = cv2.findNonZero(skin_mask)
            if coords is not None:
                rx, ry, rw, rh = cv2.boundingRect(coords)
                
                # Add padding within the ROI limits
                padding = 20
                rx_pad = max(0, rx - padding)
                ry_pad = max(0, ry - padding)
                rw_pad = min(w_box - rx_pad, rw + 2 * padding)
                rh_pad = min(h_box - ry_pad, rh + 2 * padding)
                
                # Translate crop back to original image space
                x = x_start + rx_pad
                y = y_start + ry_pad
                w = rw_pad
                h = rh_pad
                bbox = (x, y, w, h)
            else:
                bbox = (x_start, y_start, w_box, h_box)
                
            print(f"[Hand Detection] ROI Skin SUCCESS: ratio={skin_ratio:.4f}, bbox={bbox}")
            return True, 1, bbox
        else:
            # Low skin ratio inside ROI and face masked out -> return no hand.
            print(f"[Hand Detection] ROI Skin NONE: ratio={skin_ratio:.4f}")
            return False, 0, None

    except Exception as e:
        print(f"[Hand Detection] ROI Fallback Error: {e}")
        return False, 0, None


def check_hand_position(bbox, w_img, h_img):
    """
    Checks if the hand bounding box centroid is centered inside the ROI guide box.
    Returns (status: str, message: str)
    """
    if bbox is None:
        return "no_hands", "Correct your hand position"
        
    x, y, w, h = bbox
    # ROI boundaries matching frontend
    w_box = int(w_img * 0.55)
    h_box = int(h_img * 0.70)
    x_start = (w_img - w_box) // 2
    y_start = (h_img - h_box) // 2
    
    hand_cx = x + w / 2
    hand_cy = y + h / 2
    
    roi_cx = x_start + w_box / 2
    roi_cy = y_start + h_box / 2
    
    # 25% tolerance relative to ROI width/height
    tol_x = w_box * 0.25
    tol_y = h_box * 0.25
    
    if hand_cx < roi_cx - tol_x:
        return "incorrect_position", "Move hand to the right"
    elif hand_cx > roi_cx + tol_x:
        return "incorrect_position", "Move hand to the left"
    elif hand_cy < roi_cy - tol_y:
        return "incorrect_position", "Move hand down"
    elif hand_cy > roi_cy + tol_y:
        return "incorrect_position", "Move hand up"
        
    return "correct", "Hand is in center"



def preprocess_for_classifier(base64_str):
    """
    Full preprocessing pipeline:
      1. Decode base64 → OpenCV image
      2. Run hand detection
      3. Resize to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE

    Returns:
        (processed_image, has_hands: bool, hand_count: int)
    """
    cv_image = decode_base64_image(base64_str)
    if cv_image is None:
        return None, False, 0

    has_hands, hand_count, bbox = detect_hands_in_image(cv_image)
    if not has_hands:
        return None, False, hand_count

    if bbox is not None:
        x, y, w, h = bbox
        if w > 20 and h > 20: # Ensure valid crop
            cv_image = cv_image[y:y+h, x:x+w]

    img_resized = cv2.resize(cv_image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    return img_resized, True, hand_count


def extract_landmarks(cv_image):
    """
    Extracts 21 hand landmarks from an image using MediaPipe Hands.
    Returns:
        np.array of shape (63,) normalized landmarks, or None if no hand detected.
    """
    global _mp_hands_detector
    if _mp_hands_detector is None:
        return None
        
    try:
        # Convert BGR to RGB
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image_rgb)
        results = _mp_hands_detector.detect(mp_image)
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Extract landmark coordinates
            temp_coords = []
            for lm in hand_landmarks:
                temp_coords.append([lm.x, lm.y, lm.z])
            temp_coords = np.array(temp_coords)
            
            # Translate wrist to origin (landmark 0) to make translation invariant
            origin = temp_coords[0]
            temp_coords = temp_coords - origin
            
            # Scale coordinates by the max distance to make scale invariant
            # Distance from wrist (0) to middle finger MCP (9)
            scale = np.linalg.norm(temp_coords[9])
            if scale > 0:
                temp_coords = temp_coords / scale
                
            return temp_coords.flatten()
            
    except Exception as e:
        print(f"[Hand Landmark Extraction] Error: {e}")
        
    return None


def extract_landmarks_from_b64(base64_str):
    """
    Decodes base64 image and extracts landmarks.
    """
    cv_image = decode_base64_image(base64_str)
    if cv_image is None:
        return None
    return extract_landmarks(cv_image)

