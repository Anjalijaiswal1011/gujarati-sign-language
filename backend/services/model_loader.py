import os
import random
import numpy as np
from services.constants import (
    ALPHABET_MODEL_PATH, ALPHABET_LABELS_PATH,
    WORDS_MODEL_PATH, WORDS_LABELS_PATH,
    ALPHABET_LABELS, WORDS_LABELS
)

def _safe_load_keras_model(model_path):
    """Load a Keras model and retry with unsafe deserialization if needed.
    This helps with models that contain `Lambda` layers which require
    enabling unsafe deserialization in trusted environments.
    """
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        msg = str(e)
        if "Lambda" in msg or "deserial" in msg or "unsafe" in msg:
            try:
                from tensorflow import keras
                try:
                    keras.config.enable_unsafe_deserialization()
                except Exception:
                    pass
                try:
                    return tf.keras.models.load_model(model_path, safe_mode=False)
                except TypeError:
                    # In older Keras/TF versions, safe_mode is not a valid parameter
                    return tf.keras.models.load_model(model_path)
            except Exception as e2:
                raise e2
        raise

class CVZoneClassifierMock:
    """
    A mock class that mimics cvzone.ClassificationModule.Classifier
    to prevent crashes if cvzone/tensorflow are not installed,
    while still providing simulated predictions.
    """
    def __init__(self, model_path, labels_path):
        self.model_path = model_path
        self.labels_path = labels_path
        # Load labels from file if it exists, otherwise fallback to constants
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines() if line.strip()]
            except Exception:
                self.labels = []

        if not self.labels:
            if "model1" in model_path.lower() or "alphabet" in model_path.lower():
                self.labels = ALPHABET_LABELS
            else:
                self.labels = WORDS_LABELS
        self.num_classes = len(self.labels)

    def getPrediction(self, img, draw=False):
        """
        Simulates getPrediction and returns a high-confidence random predicted gesture.
        This enables full end-to-end frontend testing of matras and confirmations.
        """
        import random
        # 10% chance of returning low confidence / no gesture
        if random.random() < 0.10:
            probs = [0.0] * self.num_classes
            return probs, 0
            
        probs = [0.0] * self.num_classes
        best_idx = random.randint(0, self.num_classes - 1)
        probs[best_idx] = random.uniform(0.72, 0.98) # 72% to 98% confidence
        return probs, best_idx

class LandmarkMLPClassifier:
    """
    A classifier that runs predictions on a single frame of hand landmarks
    (flat 63 float coordinates) using a trained MLP model.
    """
    def __init__(self, model_path, labels_path):
        # Use the safe loader which retries with unsafe deserialization if needed
        self.model = _safe_load_keras_model(model_path)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            
    def getPrediction(self, landmarks_flat, draw=False):
        import numpy as np
        # Input shape: (1, 63)
        features = np.expand_dims(landmarks_flat, axis=0).astype(np.float32)
        preds = self.model.predict(features, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        probs = preds.tolist()
        return probs, best_idx

class LSTMClassifier:
    """
    A sequence classifier that runs predictions on a 30-frame window of hand landmarks
    (shape 30x63) using a trained LSTM model.
    """
    def __init__(self, model_path, labels_path):
        self.model = _safe_load_keras_model(model_path)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            
    def getPrediction(self, sequence_30x63, draw=False):
        import numpy as np
        # Input shape: (1, 30, 63)
        features = np.expand_dims(sequence_30x63, axis=0).astype(np.float32)
        preds = self.model.predict(features, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        probs = preds.tolist()
        return probs, best_idx

class CustomKerasClassifier:
    """
    A custom classifier that feeds the raw image directly into 
    our custom-trained Keras CNN model.
    """
    def __init__(self, model_path, labels_path):
        self.model = _safe_load_keras_model(model_path)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
    def getPrediction(self, img, draw=False):
        import cv2
        import numpy as np
        
        # The model was trained on 64x64 RGB images
        img_resized = cv2.resize(img, (64, 64))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Expand dims to create a batch of 1
        features = np.expand_dims(img_rgb, axis=0).astype(np.float32)
        
        preds = self.model.predict(features, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        probs = preds.tolist()
            
        return probs, best_idx

def load_classifier(mode="words"):
    """
    Loads the appropriate classifier based on the mode.
    First tries to use the new landmark-based models (MLP for alphabet, LSTM for words).
    Then falls back to the custom image-based CNN model.
    Then falls back to real CVZone Classifier if available.
    Finally falls back to a Mock/Placeholder.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Try our new MediaPipe hand landmark based models (highly recommended for speed and accuracy)
    if mode == "alphabet":
        landmarks_model_path = os.path.join(base_dir, "dataset", "models", "alphabet_landmarks_model.h5")
        landmarks_labels_path = os.path.join(base_dir, "dataset", "models", "alphabet_landmarks_labels.txt")
        if os.path.exists(landmarks_model_path) and os.path.exists(landmarks_labels_path):
            try:
                print(f"[ModelLoader] Loading Landmark MLP Classifier for {mode}")
                return LandmarkMLPClassifier(landmarks_model_path, landmarks_labels_path)
            except Exception as e:
                print(f"[ModelLoader] Error loading landmark MLP: {e}")
    else:
        lstm_model_path = os.path.join(base_dir, "dataset", "models", "words_lstm_model.h5")
        lstm_labels_path = os.path.join(base_dir, "dataset", "models", "words_lstm_labels.txt")
        if os.path.exists(lstm_model_path) and os.path.exists(lstm_labels_path):
            try:
                print(f"[ModelLoader] Loading LSTM Sequence Classifier for {mode}")
                return LSTMClassifier(lstm_model_path, lstm_labels_path)
            except Exception as e:
                print(f"[ModelLoader] Error loading LSTM model: {e}")

    # 2. Fallback: Try our custom trained image-based Keras CNN model
    if mode == "alphabet":
        custom_model_path = os.path.join(base_dir, "dataset", "models", "alphabet_model.h5")
        custom_labels_path = os.path.join(base_dir, "dataset", "models", "alphabet_labels.txt")
        # Backward compatibility fallback
        if not os.path.exists(custom_model_path):
            custom_model_path = os.path.join(base_dir, "dataset", "models", "gesture_model.h5")
            custom_labels_path = os.path.join(base_dir, "dataset", "models", "labels.txt")
    else:
        custom_model_path = os.path.join(base_dir, "dataset", "models", "words_model.h5")
        custom_labels_path = os.path.join(base_dir, "dataset", "models", "words_labels.txt")
    
    if os.path.exists(custom_model_path) and os.path.exists(custom_labels_path):
        try:
            print(f"[ModelLoader] Using Custom Keras Image CNN Classifier for {mode}")
            return CustomKerasClassifier(custom_model_path, custom_labels_path)
        except Exception as e:
            print(f"[ModelLoader] Error loading custom image model for {mode}: {e}")

    # 3. Fallback: Try old CVZone model paths
    if mode == "alphabet":
        m_path = ALPHABET_MODEL_PATH
        l_path = ALPHABET_LABELS_PATH
    else:
        m_path = WORDS_MODEL_PATH
        l_path = WORDS_LABELS_PATH

    full_m_path = os.path.join(base_dir, m_path)

    # Try loading real CVZone classifier if dependencies and files are present
    if os.path.exists(full_m_path):
        try:
            from cvzone.ClassificationModule import Classifier
            return Classifier(full_m_path, os.path.join(base_dir, l_path))
        except Exception as e:
            print(f"[ModelLoader] Could not load real CVZone classifier for {mode}: {e}")
    
    # Fallback to simulated classifier for demonstration
    print(f"[ModelLoader] Using simulated classifier for {mode}")
    return CVZoneClassifierMock(full_m_path, l_path)

# Singleton instances and modification timestamps for performance
_alphabet_classifier = None
_words_classifier = None
_alphabet_mtime = None
_words_mtime = None

def get_classifier(mode="words"):
    global _alphabet_classifier, _words_classifier
    global _alphabet_mtime, _words_mtime
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if mode == "alphabet":
        landmarks_model_path = os.path.join(base_dir, "dataset", "models", "alphabet_landmarks_model.h5")
        current_mtime = os.path.getmtime(landmarks_model_path) if os.path.exists(landmarks_model_path) else None
        
        if _alphabet_classifier is None or current_mtime != _alphabet_mtime:
            _alphabet_classifier = load_classifier("alphabet")
            _alphabet_mtime = current_mtime
        return _alphabet_classifier
    else:
        lstm_model_path = os.path.join(base_dir, "dataset", "models", "words_lstm_model.h5")
        current_mtime = os.path.getmtime(lstm_model_path) if os.path.exists(lstm_model_path) else None
        
        if _words_classifier is None or current_mtime != _words_mtime:
            _words_classifier = load_classifier("words")
            _words_mtime = current_mtime
        return _words_classifier


