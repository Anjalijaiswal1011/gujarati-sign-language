# Mapping from MediaPipe Hand Gesture labels to user-friendly English strings
# Built-in MediaPipe labels usually include: 
# None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, Love

MP_GESTURE_MAP = {
    "Open_Palm": "Open Hand",
    "Thumb_Up": "Thumb Up",
    "Victory": "Victory",
    "Love": "I Love You",
    "Pointing_Up": "Upward Point",
    "Closed_Fist": "Closed Fist",
    "Thumb_Down": "Thumb Down"
}

# Fallback/Extended mapping for custom classifier or Kaggle-based model
CUSTOM_GESTURE_MAP = {
    "OK": "OK",
    "Peace": "Peace",
    "Hello": "Hello",
    "Thank You": "Thank You"
}

def map_gesture_label(label):
    """
    Returns a clean English version of the detected gesture label.
    Prioritizes built-in MediaPipe mappings then falls back to custom ones.
    """
    if not label:
        return "Unknown"
        
    return MP_GESTURE_MAP.get(label, CUSTOM_GESTURE_MAP.get(label, label))
