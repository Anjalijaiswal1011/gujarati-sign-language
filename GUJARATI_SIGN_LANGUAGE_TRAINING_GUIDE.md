# Gujarati Sign Language (GSL) Training Guide
## Complete Implementation for Word & Action Detection

---

## 📋 Table of Contents
1. [Dataset Creation](#1-dataset-creation)
2. [Landmark Extraction](#2-landmark-extraction-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Real-Time Detection](#5-real-time-detection-pipeline)
6. [Stability & Confidence](#6-stability--confidence-management)
7. [Gujarati Translation](#7-gujarati-translation)
8. [Complete Workflow](#8-complete-workflow-diagram)

---

## 1. Dataset Creation

### 1.1 Directory Structure

```
dataset/
├── raw_images/              # Static alphabet (A-Z + special)
│   ├── A/
│   ├── B/
│   └── ... (existing)
│
├── words/                   # NEW: Static word gestures
│   ├── hello/
│   ├── thank_you/
│   ├── kem_cho/
│   ├── eat/
│   ├── drink/
│   ├── come/
│   ├── go/
│   ├── help/
│   ├── stop/
│   └── ... (30 total words)
│
├── actions/                 # NEW: Dynamic action videos
│   ├── walking/
│   ├── eating/
│   ├── drinking/
│   ├── helping/
│   ├── greeting/
│   └── ...
│
├── sequences/               # Extracted landmark sequences
│   ├── words/
│   │   ├── hello/
│   │   │   ├── user_1_seq_001.npy
│   │   │   └── ...
│   │   └── ...
│   ├── actions/
│   │   ├── walking/
│   │   │   ├── video_001.npy
│   │   │   └── ...
│   │   └── ...
│   └── landmarks.csv        # Central landmark storage
│
└── models/
    ├── alphabet_model.h5          # Static alphabet CNN
    ├── words_lstm_model.h5        # Word sequence LSTM
    ├── action_model.h5            # Action CNN+LSTM
    ├── alphabet_labels.txt
    ├── words_labels.txt
    ├── action_labels.txt
    └── training_progress.json
```

### 1.2 Dataset Requirements

#### For Word Gestures (Static):
- **Samples per word**: 500-2000 images
- **Multiple users**: 5-10+ different signers
- **Lighting variations**: 3+ different lighting conditions
- **Backgrounds**: Indoor, outdoor, different settings
- **Hand angles**: Front, side, different hand positions
- **Resolution**: 640×480 minimum

#### For Action Gestures (Dynamic - Video):
- **Samples per action**: 100-500 videos
- **Video length**: 1-3 seconds (30-90 frames @ 30fps)
- **Multiple users**: 3-8+ different signers
- **FPS**: 30 frames per second
- **Resolution**: 640×480 minimum

---

## 2. Landmark Extraction Pipeline

### 2.1 MediaPipe Hand Landmarks (21 points)

Each hand produces 21 landmarks:
```
0:  Wrist
1-4:  Thumb (base, middle, pip, tip)
5-8:  Index (mcp, pip, dip, tip)
9-12: Middle (mcp, pip, dip, tip)
13-16: Ring (mcp, pip, dip, tip)
17-20: Pinky (mcp, pip, dip, tip)

Per landmark: (x, y, z, confidence)
Total: 21 landmarks × 4 values = 84 features per frame
```

### 2.2 Extraction Pipeline

```
Video Frame
    ↓
[MediaPipe Hands Detector]
    ↓
Extract 21 landmarks × 4 (x, y, z, confidence)
    ↓
Normalize landmarks (to [0, 1] range)
    ↓
Smooth across frames (optional: temporal smoothing)
    ↓
Store as numpy array: shape (sequence_length, 84)
    ↓
Save: landmark_sequence.npy
```

### 2.3 Landmark Normalization

```python
# Normalize to bounding box center
def normalize_landmarks(landmarks):
    # landmarks shape: (21, 4)
    
    # Get hand bounding box
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Center
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Normalize to [-1, 1] range
    hand_size = max(x_max - x_min, y_max - y_min)
    landmarks_norm = landmarks.copy()
    landmarks_norm[:, 0] = (landmarks[:, 0] - x_center) / (hand_size / 2)
    landmarks_norm[:, 1] = (landmarks[:, 1] - y_center) / (hand_size / 2)
    landmarks_norm[:, 2] = landmarks[:, 2] / (hand_size / 2)  # Z
    
    return landmarks_norm
```

### 2.4 Sequence Storage Format

```python
# For a video sequence (30 frames):
sequence = np.zeros((30, 84))  # 30 frames, 84 features each

# Fill with normalized landmarks
for frame_idx, landmarks_list in enumerate(all_frames):
    if len(landmarks_list) > 0:
        sequence[frame_idx] = landmarks_list[0].flatten()  # First hand

np.save('hello_user1_seq1.npy', sequence)
```

---

## 3. Model Architecture

### 3.1 Model for Static Words (CNN only)

```
Input: 300×300 RGB image
    ↓
Rescaling: /255
    ↓
Conv2D(32, 3×3) → ReLU → MaxPool2D(2×2)
    ↓
Conv2D(64, 3×3) → ReLU → MaxPool2D(2×2)
    ↓
Conv2D(128, 3×3) → ReLU → MaxPool2D(2×2)
    ↓
Flatten
    ↓
Dense(256, ReLU) → Dropout(0.3)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(30, Softmax)  ← 30 word classes
    ↓
Output: Word prediction
```

**Architecture Details**:
- Input: (300, 300, 3)
- Trainable parameters: ~1.2M
- Training: 20-50 epochs, batch size 32
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy

---

### 3.2 Model for Word Sequences (LSTM)

```
Input: 30-frame sequence (30, 84)
    ↓
Dense(128, ReLU)
    ↓
Bidirectional LSTM(256 units) → Return sequences
    ↓
Bidirectional LSTM(128 units) → Return sequences
    ↓
Dense(64, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Bidirectional LSTM(64 units) → Return final state only
    ↓
Dense(64, ReLU) → Dropout(0.2)
    ↓
Dense(30, Softmax)  ← 30 word classes
    ↓
Output: Word prediction
```

**Architecture Details**:
- Input: (sequence_length=30, features=84)
- Total trainable parameters: ~350K
- Training: 100 epochs, batch size 16
- Early stopping: patience=15
- ReduceLR: factor=0.5, patience=5
- Data augmentation: 3× original dataset size

**Key Features**:
- Bidirectional LSTM captures past & future context
- Multiple layers learn hierarchical temporal patterns
- Batch normalization stabilizes training
- Dropout prevents overfitting

---

### 3.3 Model for Action Detection (CNN+LSTM)

```
Input: 30-frame sequence (30, 84)  OR  (30, 3, 224, 224) if using CNN features
    ↓
[Option 1: Landmark-based]
    Dense(256, ReLU) → BatchNorm
        ↓
    Bidirectional LSTM(256) → Return sequences
        ↓
    Bidirectional LSTM(128) → Return final
        ↓
    Dense(64, ReLU) → Dropout(0.3)
        ↓
    Dense(10, Softmax)  ← 10 action classes
    
[Option 2: Image-based CNN+LSTM]
    TimeDistributed(Conv2D(32, 3×3)) → ReLU
        ↓
    TimeDistributed(MaxPool2D) 
        ↓
    TimeDistributed(Conv2D(64, 3×3)) → ReLU
        ↓
    TimeDistributed(Flatten)
        ↓
    Bidirectional LSTM(256) → Return sequences
        ↓
    Bidirectional LSTM(128) → Return final
        ↓
    Dense(64, ReLU) → Dropout(0.3)
        ↓
    Dense(10, Softmax)  ← 10 action classes
```

**Recommended**: Use Option 1 (landmark-based) for:
- Faster inference
- Less GPU memory
- More robust to background changes
- Better for real-time deployment

---

## 4. Training Pipeline

### 4.1 Data Preparation

**Step 1: Organize raw images/videos**
```bash
# Words (static images)
mkdir -p dataset/words/hello
mkdir -p dataset/words/thank_you
# ... (create for all 30 words)

# Actions (video folders)
mkdir -p dataset/actions/walking
mkdir -p dataset/actions/eating
# ... (create for all actions)
```

**Step 2: Extract landmarks from words (static)**
```python
# Run: python scripts/extract_landmarks_for_sequences.py --mode=words
# This processes all images in dataset/words/*/ 
# and saves sequences to dataset/sequences/words/
```

**Step 3: Extract landmarks from action videos**
```python
# Run: python scripts/extract_landmarks_for_sequences.py --mode=actions
# This processes all videos in dataset/actions/*/
# and saves sequences to dataset/sequences/actions/
```

### 4.2 Training Split

```
Total samples: 100%
├── Training: 70%     (used for gradient updates)
├── Validation: 15%   (used for hyperparameter tuning)
└── Testing: 15%      (used for final evaluation only)
```

### 4.3 Training Configuration

```python
TRAINING_CONFIG = {
    # Alphabet (CNN, static images)
    "alphabet": {
        "epochs": 50,
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "early_stopping_patience": 10,
    },
    
    # Words (LSTM, sequences)
    "words": {
        "epochs": 100,
        "batch_size": 16,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "early_stopping_patience": 15,
        "augmentation_factor": 3,  # 3× data augmentation
    },
    
    # Actions (CNN+LSTM, video sequences)
    "actions": {
        "epochs": 100,
        "batch_size": 16,
        "optimizer": "adam",
        "learning_rate": 0.0005,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "early_stopping_patience": 15,
        "augmentation_factor": 3,
    }
}
```

### 4.4 Training Callbacks

1. **Early Stopping**: Stop if val_accuracy doesn't improve for 15 epochs
2. **ReduceLROnPlateau**: Reduce learning rate by 0.5 if loss plateaus
3. **ModelCheckpoint**: Save best model based on val_accuracy
4. **TrainingProgressCallback**: Log progress to JSON for monitoring

---

## 5. Real-Time Detection Pipeline

### 5.1 Detection Pipeline Architecture

```
Webcam/Camera
    ↓
[Frame Capture @ 30fps]
    ↓
[MediaPipe Hand Detection]
    ↓
[Landmark Extraction] → (21 landmarks × 4 = 84 features)
    ↓
[Sequence Buffering] → Keep rolling window of 30 frames
    ↓
[Model Inference]
├─ If 30 frames in buffer → Run through LSTM
├─ Confidence > 90%?
├─ Same prediction for > 15 consecutive frames?
│   ↓ YES
│   ↓
│   [Gesture Confirmed]
│       ↓
│   [Gujarati Translation]
│       ↓
│   [Output to Frontend]
│       ↓
│   [Cooldown Period: 2 seconds]
│
└─ NO: Continue buffering
```

### 5.2 Real-Time Sequence Buffer

```python
class SequenceBuffer:
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.buffer = []
    
    def add_frame(self, landmarks):
        """Add normalized landmarks to buffer."""
        self.buffer.append(landmarks)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
    
    def is_ready(self):
        """Check if buffer has enough frames."""
        return len(self.buffer) == self.max_length
    
    def get_sequence(self):
        """Return buffer as numpy array."""
        return np.array(self.buffer)
    
    def clear(self):
        """Reset buffer."""
        self.buffer = []
```

---

## 6. Stability & Confidence Management

### 6.1 Confidence Threshold

```python
def validate_prediction(confidence, threshold=0.90):
    """
    confidence: float (0-1)
    threshold: minimum confidence to accept
    """
    return confidence >= threshold
```

**Recommended thresholds**:
- Alphabet gestures: 0.85 (more lenient for real-time)
- Word sequences: 0.90 (higher requirement for LSTM)
- Action sequences: 0.90 (higher requirement for temporal)

### 6.2 Stable Frame Checking

```python
class StabilityFilter:
    def __init__(self, stability_window=15, confidence_threshold=0.90):
        """
        stability_window: min consecutive frames with same prediction
        confidence_threshold: min confidence for acceptance
        """
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        
        self.prediction_history = []
        self.confidence_history = []
    
    def add_prediction(self, label, confidence):
        """Add a prediction to tracking."""
        self.prediction_history.append(label)
        self.confidence_history.append(confidence)
        
        # Keep only last 30 frames
        if len(self.prediction_history) > 30:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
    
    def is_stable(self):
        """Check if prediction is stable and confident."""
        if len(self.prediction_history) < self.stability_window:
            return False, None, 0.0
        
        # Get last N predictions
        recent = self.prediction_history[-self.stability_window:]
        recent_confidence = self.confidence_history[-self.stability_window:]
        
        # Check if all recent predictions are same
        if len(set(recent)) != 1:
            return False, None, 0.0
        
        # Check confidence average
        avg_confidence = np.mean(recent_confidence)
        
        if avg_confidence < self.confidence_threshold:
            return False, None, avg_confidence
        
        # Stable!
        return True, recent[0], avg_confidence
```

### 6.3 Prediction Smoothing (Temporal Smoothing)

```python
class TemporalSmoother:
    def __init__(self, window_size=5):
        """Smooth predictions using median filter."""
        self.window_size = window_size
        self.predictions = []
    
    def smooth(self, prediction):
        """Apply temporal smoothing."""
        self.predictions.append(prediction)
        
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        
        # Return most frequent prediction in window
        from collections import Counter
        counter = Counter(self.predictions)
        return counter.most_common(1)[0][0]
```

### 6.4 Cooldown Between Predictions

```python
import time

class CooldownManager:
    def __init__(self, cooldown_seconds=2.0):
        """Prevent duplicate outputs within cooldown period."""
        self.cooldown = cooldown_seconds
        self.last_prediction_time = 0
        self.last_prediction = None
    
    def can_predict(self, prediction):
        """Check if enough time passed since last prediction."""
        current_time = time.time()
        
        # Different prediction: always allow
        if prediction != self.last_prediction:
            return True
        
        # Same prediction: check cooldown
        if (current_time - self.last_prediction_time) >= self.cooldown:
            return True
        
        return False
    
    def record_prediction(self, prediction):
        """Record this prediction."""
        self.last_prediction = prediction
        self.last_prediction_time = time.time()
```

### 6.5 Complete Stability Pipeline

```python
def process_real_time_prediction(label, confidence, sequence):
    """
    Complete pipeline with all stability checks.
    """
    # 1. Confidence check
    if confidence < CONFIDENCE_THRESHOLD:
        return None, f"Low confidence: {confidence:.2%}"
    
    # 2. Add to stability filter
    is_stable, stable_label, avg_conf = stability_filter.is_stable()
    
    if not is_stable:
        return None, "Predicting (need more consistent frames)"
    
    # 3. Temporal smoothing
    smoothed_label = temporal_smoother.smooth(stable_label)
    
    # 4. Cooldown check (prevent duplicates)
    if not cooldown_manager.can_predict(smoothed_label):
        return None, "Still in cooldown period"
    
    # 5. Record prediction
    cooldown_manager.record_prediction(smoothed_label)
    
    # SUCCESS!
    return smoothed_label, "Gesture confirmed"
```

---

## 7. Gujarati Translation

### 7.1 Word Mapping

All detected words map directly to Gujarati:

```python
WORD_TO_GUJARATI = {
    "hello": "નમસ્તે",
    "thank_you": "આભાર",
    "yes": "હા",
    "no": "ના",
    "eat": "ખાઓ",
    "drink": "પીઓ",
    "come": "આવો",
    "go": "જાઓ",
    "help": "મદદ",
    "stop": "રોકો",
    # ... 20 more words
}
```

### 7.2 Translation Logic

```python
def translate_gesture_to_gujarati(detected_label):
    """
    Convert detected gesture label to Gujarati text.
    
    Args:
        detected_label: string (e.g., "hello")
    
    Returns:
        gujarati_text: string (e.g., "નમસ્તે")
    """
    return WORD_TO_GUJARATI.get(detected_label, detected_label)
```

### 7.3 Post-Processing Phrases

```python
# Auto-complete common phrases
PHRASE_EXTENSIONS = {
    "hello": "નમસ્તે, શું હાલ છે?",      # "Hello, how are you?"
    "thank_you": "બહુ આભાર",             # "Thank you very much"
    "come": "અહીં આવો",                  # "Come here"
    "go": "જઈ આવો",                     # "Go on"
}
```

---

## 8. Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   GUJARATI SIGN LANGUAGE SYSTEM             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE (Training)                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Dataset Collection (Videos, Images)                        │
│         ↓                                                     │
│  Extract Landmarks (MediaPipe)                              │
│         ↓                                                     │
│  Store Sequences (NP arrays)                                │
│         ↓                                                     │
│  Data Augmentation (3× size)                                │
│         ↓                                                     │
│  Train Models:                                              │
│    - Alphabet CNN (50 epochs)                               │
│    - Words LSTM (100 epochs)                                │
│    - Action CNN+LSTM (100 epochs)                           │
│         ↓                                                     │
│  Evaluate & Save Best Models                                │
│         ↓                                                     │
│  Save Labels, Thresholds, Config                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   ONLINE (Real-Time Inference)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Webcam Input (30 fps)                                      │
│         ↓                                                     │
│  MediaPipe Hand Detection                                   │
│         ↓                                                     │
│  Extract Landmarks (21 points × 4 features = 84)            │
│         ↓                                                     │
│  Add to Sequence Buffer (rolling 30-frame window)           │
│         ↓                                                     │
│  Buffer Full?                                               │
│    - NO: Continue buffering                                 │
│    - YES: Send to Model                                     │
│         ↓                                                     │
│  LSTM Inference → Label + Confidence                        │
│         ↓                                                     │
│  Stability Checks:                                          │
│    1. Confidence > 0.90?                                    │
│    2. Same prediction for 15 frames?                        │
│    3. Not in cooldown?                                      │
│         ↓                                                     │
│  All Checks Pass?                                           │
│    - NO: Continue monitoring                                │
│    - YES: Translate to Gujarati                             │
│         ↓                                                     │
│  Gujarati Translation (dict lookup)                         │
│         ↓                                                     │
│  Display to User:                                           │
│    - English gesture label                                  │
│    - Gujarati translation                                   │
│    - Confidence score                                       │
│         ↓                                                     │
│  Activate Cooldown (2 seconds)                              │
│         ↓                                                     │
│  Back to Webcam Input                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Checklist

- [ ] Create dataset folder structure
- [ ] Collect word gesture images (500-2000 per word)
- [ ] Collect action gesture videos (100-500 per action)
- [ ] Implement landmark extraction script
- [ ] Extract landmarks from all word/action data
- [ ] Implement sequence augmentation
- [ ] Train alphabet CNN (50 epochs)
- [ ] Train word LSTM (100 epochs)
- [ ] Train action CNN+LSTM (100 epochs)
- [ ] Evaluate all models (70/15/15 split)
- [ ] Implement stability filter
- [ ] Implement cooldown manager
- [ ] Implement real-time pipeline
- [ ] Test with live webcam
- [ ] Optimize inference speed
- [ ] Deploy to production

---

## 10. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Alphabet Accuracy** | >95% | Single frames, fast |
| **Word Accuracy** | >92% | Sequence-based |
| **Action Accuracy** | >90% | Dynamic gestures |
| **Inference Speed** | <100ms | Real-time capable |
| **FPS** | 25+ fps | Smooth video |
| **False Positive Rate** | <1% | Good stability |
| **Confidence Threshold** | 0.90 | 90% confidence |

---

## 11. Quick Start Commands

```bash
# 1. Extract landmarks from existing images/videos
python scripts/extract_landmarks_for_sequences.py --mode=words
python scripts/extract_landmarks_for_sequences.py --mode=actions

# 2. Train models
python scripts/train_alphabet_model.py
python scripts/train_words_lstm_model.py
python scripts/train_action_model.py

# 3. Evaluate models
python scripts/evaluate_models.py

# 4. Start real-time detection
python scripts/real_time_detection.py

# 5. Generate dataset (if data exists)
python scripts/action_dataset_generator.py
```

---

## Summary

This guide provides:
1. ✅ Dataset structure for 30 Gujarati words + actions
2. ✅ Landmark extraction pipeline (21 points × 4 features)
3. ✅ CNN for static alphabet
4. ✅ LSTM for word sequences (30-frame windows)
5. ✅ CNN+LSTM for action detection
6. ✅ Confidence thresholds (0.90)
7. ✅ Stability filtering (15 consecutive frames)
8. ✅ Real-time pipeline with WebSockets
9. ✅ Direct word-to-Gujarati translation
10. ✅ Production-ready stability management

**Next Steps**: Implement the scripts in the `scripts/` folder using the architectures described above.
