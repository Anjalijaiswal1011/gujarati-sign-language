import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from .preprocessing import preprocess_for_classifier, extract_landmarks_from_b64
from .model_loader import get_classifier, LandmarkMLPClassifier, LSTMClassifier, CVZoneClassifierMock, CustomKerasClassifier
from .constants import (
    ALPHABET_LABELS, WORDS_LABELS, 
    CONFIDENCE_THRESHOLD, WORD_CONFIDENCE_THRESHOLD, MODE_ALPHABET
)
from services.translation_service import translate_to_gujarati
from services.nlp_service import NLPProcessor

def process_gesture_frame(image_b64, mode="words", threshold=None):
    """
    Main pipeline for processing single frames (primarily static alphabet spelling).
    """
    try:
        classifier = get_classifier(mode)
        is_mock = isinstance(classifier, CVZoneClassifierMock)
        conf_thresh = threshold if threshold is not None else CONFIDENCE_THRESHOLD

        # 1. If the classifier is a Landmark MLP classifier, extract and feed landmarks
        if isinstance(classifier, LandmarkMLPClassifier):
            landmarks = extract_landmarks_from_b64(image_b64)
            if landmarks is None:
                return {
                    "success": False,
                    "status": "no_hands",
                    "message": "No hands detected in image",
                    "label": "No gesture detected",
                    "confidence": 0
                }
            
            prediction, index = classifier.getPrediction(landmarks)
            confidence = float(np.max(prediction)) * 100
            labels = classifier.labels
        
        # 2. Otherwise, fall back to image-based CNN
        else:
            from .preprocessing import decode_base64_image, detect_hands_in_image, check_hand_position
            cv_image = decode_base64_image(image_b64)
            if cv_image is not None:
                has_hands, hand_count, bbox = detect_hands_in_image(cv_image)
                if has_hands and bbox is not None:
                    h_img, w_img, _ = cv_image.shape
                    pos_status, pos_msg = check_hand_position(bbox, w_img, h_img)
                    if pos_status == "incorrect_position":
                        w_box = int(w_img * 0.55)
                        if bbox[2] != w_box: # Avoid blocking if it's the fallback center ROI box itself
                            # Log the position warning, but do NOT block prediction
                            print(f"[GestureService] Hand position suggestion: {pos_msg}")

            processed_image, has_hands, hand_count = preprocess_for_classifier(image_b64)
            
            if not has_hands:
                return {
                    "success": False,
                    "status": "no_hands",
                    "message": "No hands detected in image",
                    "label": "No gesture detected",
                    "confidence": 0,
                    "hand_count": hand_count
                }

            if processed_image is None:
                return {
                    "success": False,
                    "status": "error",
                    "message": "Failed to process image"
                }

            prediction, index = classifier.getPrediction(processed_image, draw=False)
            confidence = float(np.max(prediction)) * 100
            
            if isinstance(classifier, CustomKerasClassifier):
                labels = classifier.labels
            else:
                labels = ALPHABET_LABELS if mode == MODE_ALPHABET else WORDS_LABELS

        # 3. Process label and translation
        if 0 <= index < len(labels) and confidence >= conf_thresh:
            predicted_label = labels[index]
            
            # Translate to Gujarati
            gujarati_label = translate_to_gujarati(predicted_label)
            
            print(f"[GestureService] Static Prediction: {predicted_label} ({confidence:.1f}%) -> {gujarati_label}")
            return {
                "success": True,
                "status": "success",
                "label": predicted_label,
                "gujarati": gujarati_label,
                "confidence": round(confidence, 2),
                "message": "Detected",
                "is_mock": is_mock
            }
        else:
            print(f"[GestureService] Low confidence: {confidence:.1f}%")
            return {
                "success": False,
                "status": "low_confidence",
                "message": f"Low confidence ({confidence:.1f}%) or unknown gesture",
                "label": labels[index] if (0 <= index < len(labels)) else "Unknown",
                "confidence": round(confidence, 2)
            }

    except Exception as e:
        print(f"[GestureService] Error: {e}")
        return {
            "success": False,
            "status": "error",
            "message": str(e)
        }

def process_gesture_sequence(landmarks_sequence, mode="words", threshold=None):
    """
    Processes a sliding sequence of 30 frames of landmarks.
    """
    try:
        classifier = get_classifier(mode)
        is_mock = isinstance(classifier, CVZoneClassifierMock)
        conf_thresh = threshold if threshold is not None else WORD_CONFIDENCE_THRESHOLD

        if not isinstance(classifier, LSTMClassifier):
            # If no LSTM model is loaded, fall back to mock or notify
            return {
                "success": False,
                "status": "model_not_lstm",
                "message": "LSTM model not loaded"
            }

        prediction, index = classifier.getPrediction(landmarks_sequence)
        confidence = float(np.max(prediction)) * 100
        labels = classifier.labels

        if 0 <= index < len(labels) and confidence >= conf_thresh:
            predicted_label = labels[index]
            
            # Translate to Gujarati
            gujarati_label = translate_to_gujarati(predicted_label)
            
            print(f"[GestureService] LSTM Sequence Prediction: {predicted_label} ({confidence:.1f}%) -> {gujarati_label}")
            return {
                "success": True,
                "status": "success",
                "label": predicted_label,
                "gujarati": gujarati_label,
                "confidence": round(confidence, 2),
                "message": "Detected",
                "is_mock": is_mock
            }
        else:
            return {
                "success": False,
                "status": "low_confidence",
                "message": f"Low confidence ({confidence:.1f}%)",
                "label": labels[index] if (0 <= index < len(labels)) else "Unknown",
                "confidence": round(confidence, 2)
            }

    except Exception as e:
        print(f"[GestureService] Sequence Error: {e}")
        return {
            "success": False,
            "status": "error",
            "message": str(e)
        }
