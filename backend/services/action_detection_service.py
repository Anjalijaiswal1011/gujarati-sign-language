"""
Action Detection Service
=========================
Real-time action/dynamic gesture detection using trained CNN+LSTM model.

Pipeline:
  Webcam → MediaPipe Landmarks → Sequence Buffer → LSTM Model → Prediction → Translation

Usage:
    from services.action_detection_service import ActionDetectionService
    
    service = ActionDetectionService(mode="balanced")
    frames = [frame1, frame2, ..., frame30]  # List of MediaPipe landmarks
    result = service.detect_action(frames)
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Any, List

from services.landmark_extraction import initialize_landmarks_detector, normalize_landmarks
from services.gesture_stability_filter import create_stability_pipeline
from services.constants import GUJARATI_WORD_MAP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "dataset", "models")
ACTION_MODEL_PATH = os.path.join(MODELS_DIR, "action_model.h5")
ACTION_LABELS_PATH = os.path.join(MODELS_DIR, "action_labels.txt")


class ActionDetectionService:
    """
    Detect dynamic actions from video sequences.
    
    Handles:
    - Sequence buffering (rolling 30-frame window)
    - Model inference
    - Stability filtering
    - Gujarati translation
    """
    
    def __init__(self, mode: str = "balanced", confidence_threshold: float = 0.90):
        """
        Initialize action detection service.
        
        Args:
            mode: stability filter preset ("aggressive", "balanced", "conservative")
            confidence_threshold: minimum confidence for action detection
        """
        self.sequence_length = 30
        self.feature_dim = 84
        
        self.model = None
        self.action_labels = []
        self.load_model()
        
        self.stability_pipeline = create_stability_pipeline(mode)
        self.confidence_threshold = confidence_threshold
        
        # Sequence buffer
        self.frame_buffer = []
        
        # Statistics
        self.detection_count = 0
        self.last_detection = None
    
    def load_model(self) -> bool:
        """Load trained action detection model."""
        try:
            if not os.path.exists(ACTION_MODEL_PATH):
                print(f"[Action Service] Model not found: {ACTION_MODEL_PATH}")
                return False
            
            from services.model_loader import _safe_load_keras_model
            self.model = _safe_load_keras_model(ACTION_MODEL_PATH)
            print(f"[Action Service] Model loaded from {ACTION_MODEL_PATH}")
            
            # Load labels
            if os.path.exists(ACTION_LABELS_PATH):
                with open(ACTION_LABELS_PATH, 'r') as f:
                    self.action_labels = [line.strip() for line in f if line.strip()]
                print(f"[Action Service] Loaded {len(self.action_labels)} action labels")
            
            return True
        
        except Exception as e:
            print(f"[Action Service] Error loading model: {e}")
            return False
    
    def add_frame_landmarks(self, landmarks: np.ndarray):
        """
        Add frame landmarks to buffer.
        
        Args:
            landmarks: normalized landmarks (84,) for one frame
        """
        if landmarks is None or len(landmarks) != self.feature_dim:
            return
        
        self.frame_buffer.append(landmarks)
        
        # Keep rolling window of sequence_length frames
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
    
    def is_buffer_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.frame_buffer) == self.sequence_length
    
    def get_buffer_status(self) -> str:
        """Get human-readable buffer status."""
        return f"{len(self.frame_buffer)}/{self.sequence_length} frames"
    
    def detect_action(self, sequence: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect action from sequence.
        
        Args:
            sequence: optional sequence to use instead of buffer
        
        Returns:
            {
                "success": bool,
                "action": str or None,
                "confidence": float,
                "status": str,
                "gujarati": str or None,
                "is_stable": bool,
                "buffer_status": str
            }
        """
        if self.model is None:
            return {
                "success": False,
                "action": None,
                "confidence": 0,
                "status": "Model not loaded",
                "gujarati": None,
                "is_stable": False,
                "buffer_status": self.get_buffer_status()
            }
        
        # Use provided sequence or buffer
        if sequence is None:
            if not self.is_buffer_ready():
                return {
                    "success": False,
                    "action": None,
                    "confidence": 0,
                    "status": f"Buffering frames... {self.get_buffer_status()}",
                    "gujarati": None,
                    "is_stable": False,
                    "buffer_status": self.get_buffer_status()
                }
            
            sequence = np.array(self.frame_buffer)
        else:
            sequence = np.array(sequence)
        
        # Ensure correct shape
        if sequence.shape != (self.sequence_length, self.feature_dim):
            return {
                "success": False,
                "action": None,
                "confidence": 0,
                "status": f"Invalid sequence shape: {sequence.shape}",
                "gujarati": None,
                "is_stable": False,
                "buffer_status": self.get_buffer_status()
            }
        
        try:
            # Inference
            predictions = self.model.predict(np.expand_dims(sequence, 0), verbose=0)
            confidence = float(np.max(predictions[0]))
            class_idx = int(np.argmax(predictions[0]))
            
            if class_idx >= len(self.action_labels):
                return {
                    "success": False,
                    "action": None,
                    "confidence": confidence,
                    "status": f"Invalid class index: {class_idx}",
                    "gujarati": None,
                    "is_stable": False,
                    "buffer_status": self.get_buffer_status()
                }
            
            action_label = self.action_labels[class_idx]
            
            # Apply stability filtering
            should_output, output_label, filter_status = self.stability_pipeline.process_prediction(
                action_label, confidence
            )
            
            if should_output and output_label:
                # Translate to Gujarati
                gujarati = self._translate_to_gujarati(output_label)
                
                self.detection_count += 1
                self.last_detection = {
                    "action": output_label,
                    "gujarati": gujarati,
                    "confidence": confidence
                }
                
                return {
                    "success": True,
                    "action": output_label,
                    "confidence": confidence,
                    "status": filter_status,
                    "gujarati": gujarati,
                    "is_stable": True,
                    "buffer_status": self.get_buffer_status()
                }
            else:
                return {
                    "success": False,
                    "action": action_label,
                    "confidence": confidence,
                    "status": filter_status,
                    "gujarati": None,
                    "is_stable": False,
                    "buffer_status": self.get_buffer_status()
                }
        
        except Exception as e:
            return {
                "success": False,
                "action": None,
                "confidence": 0,
                "status": f"Inference error: {e}",
                "gujarati": None,
                "is_stable": False,
                "buffer_status": self.get_buffer_status()
            }
    
    def _translate_to_gujarati(self, action_label: str) -> str:
        """Translate action label to Gujarati."""
        # Try direct mapping first
        if action_label in GUJARATI_WORD_MAP:
            return GUJARATI_WORD_MAP[action_label]
        
        # Try with underscores
        if action_label.replace(" ", "_") in GUJARATI_WORD_MAP:
            return GUJARATI_WORD_MAP[action_label.replace(" ", "_")]
        
        # Fallback: return original
        return action_label
    
    def reset_buffer(self):
        """Clear sequence buffer."""
        self.frame_buffer.clear()
    
    def reset_filter(self):
        """Reset stability filter."""
        self.stability_pipeline.reset()
    
    def reset_all(self):
        """Reset buffer and filter."""
        self.reset_buffer()
        self.reset_filter()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "detections": self.detection_count,
            "last_detection": self.last_detection,
            "buffer_size": len(self.frame_buffer),
            "buffer_status": self.get_buffer_status(),
            "model_loaded": self.model is not None,
            "num_actions": len(self.action_labels),
            "action_labels": self.action_labels
        }


class SequenceBuffer:
    """
    Efficient rolling buffer for managing frame sequences.
    Optimized for real-time streaming scenarios.
    """
    
    def __init__(self, max_length: int = 30):
        """
        Args:
            max_length: maximum sequence length
        """
        self.max_length = max_length
        self.buffer = []
    
    def add(self, frame: np.ndarray) -> bool:
        """
        Add frame to buffer.
        
        Args:
            frame: feature vector (should be 84-dim for landmarks)
        
        Returns:
            True if buffer is now full
        """
        if frame is not None:
            self.buffer.append(frame)
            if len(self.buffer) > self.max_length:
                self.buffer.pop(0)
        
        return self.is_full()
    
    def is_full(self) -> bool:
        """Check if buffer has reached max length."""
        return len(self.buffer) == self.max_length
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """Get buffer as numpy array if full."""
        if self.is_full():
            return np.array(self.buffer)
        return None
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def progress(self) -> float:
        """Get fill progress (0-1)."""
        return len(self.buffer) / self.max_length


if __name__ == "__main__":
    # Demo: Test action detection service
    print("=== Action Detection Service Demo ===\n")
    
    service = ActionDetectionService(mode="balanced")
    print(f"Model loaded: {service.model is not None}")
    print(f"Actions: {service.action_labels}\n")
    
    # Simulate buffering frames
    print("Simulating frame buffering...")
    for i in range(35):
        # Create dummy landmarks
        dummy_landmarks = np.random.randn(84)
        service.add_frame_landmarks(dummy_landmarks)
        
        # Try detection every 5 frames after buffer is full
        if service.is_buffer_ready() and i % 5 == 0:
            result = service.detect_action()
            print(f"Frame {i}: {result['status']}")
    
    print(f"\nStats: {service.get_stats()}")
