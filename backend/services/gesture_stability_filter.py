"""
Gesture Stability Filter
=========================
Manages confidence thresholds, stable frame detection, temporal smoothing,
and cooldown periods to prevent false positives and ensure accurate real-time detection.

Components:
  - ConfidenceValidator: Check minimum confidence thresholds
  - StabilityFilter: Require N consecutive frames with same prediction
  - TemporalSmoother: Smooth predictions using median filter
  - CooldownManager: Prevent duplicate outputs within time window
  - GestureStabilityPipeline: Complete stability checking pipeline
"""

import time
import numpy as np
from collections import Counter, deque
from typing import Tuple, Optional


class ConfidenceValidator:
    """Validate prediction confidence against threshold."""
    
    def __init__(self, threshold: float = 0.90):
        """
        Args:
            threshold: minimum confidence (0-1) to accept prediction
        """
        self.threshold = threshold
    
    def is_valid(self, confidence: float) -> bool:
        """Check if confidence exceeds threshold."""
        return confidence >= self.threshold
    
    def get_status(self, confidence: float) -> Tuple[bool, str]:
        """Get validation status with explanation."""
        if confidence >= self.threshold:
            return True, f"High confidence: {confidence:.2%}"
        else:
            return False, f"Low confidence: {confidence:.2%} (need {self.threshold:.2%})"


class StabilityFilter:
    """
    Ensure prediction is stable across multiple consecutive frames.
    Prevents temporary/random fluctuations from being recognized as valid gestures.
    """
    
    def __init__(self, stability_window: int = 15, confidence_threshold: float = 0.90):
        """
        Args:
            stability_window: minimum consecutive frames with same prediction
            confidence_threshold: minimum average confidence for stability
        """
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        
        self.prediction_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
    
    def add_prediction(self, label: str, confidence: float):
        """Record a new prediction."""
        self.prediction_history.append(label)
        self.confidence_history.append(confidence)
    
    def is_stable(self) -> Tuple[bool, Optional[str], float]:
        """
        Check if prediction is stable.
        
        Returns:
            (is_stable, label, avg_confidence)
        """
        if len(self.prediction_history) < self.stability_window:
            return False, None, 0.0
        
        # Get last N predictions
        recent_predictions = list(self.prediction_history)[-self.stability_window:]
        recent_confidences = list(self.confidence_history)[-self.stability_window:]
        
        # Check if all recent predictions are the same
        if len(set(recent_predictions)) != 1:
            # Multiple different labels in stability window
            return False, None, 0.0
        
        # Calculate average confidence
        avg_confidence = np.mean(recent_confidences)
        
        # Check confidence threshold
        if avg_confidence < self.confidence_threshold:
            return False, recent_predictions[0], avg_confidence
        
        # Stable!
        stable_label = recent_predictions[0]
        return True, stable_label, avg_confidence
    
    def reset(self):
        """Clear prediction history."""
        self.prediction_history.clear()
        self.confidence_history.clear()
    
    def get_status(self) -> str:
        """Get human-readable status."""
        history_len = len(self.prediction_history)
        if history_len == 0:
            return "No predictions yet"
        
        recent = list(self.prediction_history)[-5:]
        counter = Counter(recent)
        most_common = counter.most_common(1)[0]
        
        return f"Recent: {most_common[0]} ({most_common[1]}x) | Buffer: {history_len}/{self.stability_window}"


class TemporalSmoother:
    """
    Apply temporal smoothing to predictions using median filtering.
    Reduces noise from individual frame predictions.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: size of sliding window for median filtering
        """
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)
    
    def smooth(self, prediction: str) -> str:
        """
        Apply temporal smoothing and return smoothed prediction.
        
        Args:
            prediction: raw prediction label
        
        Returns:
            smoothed_prediction: most common prediction in window
        """
        self.prediction_window.append(prediction)
        
        if len(self.prediction_window) < self.window_size:
            return prediction
        
        # Return most frequent prediction in window
        counter = Counter(self.prediction_window)
        smoothed = counter.most_common(1)[0][0]
        
        return smoothed
    
    def reset(self):
        """Clear prediction window."""
        self.prediction_window.clear()


class CooldownManager:
    """
    Prevent duplicate outputs within a cooldown period.
    Once a gesture is detected, wait N seconds before allowing same gesture again.
    """
    
    def __init__(self, cooldown_seconds: float = 2.0):
        """
        Args:
            cooldown_seconds: minimum time between same predictions
        """
        self.cooldown = cooldown_seconds
        self.last_prediction = None
        self.last_prediction_time = 0
    
    def can_predict(self, prediction: str) -> Tuple[bool, str]:
        """
        Check if a prediction can be output now.
        
        Returns:
            (can_predict, reason)
        """
        current_time = time.time()
        time_since_last = current_time - self.last_prediction_time
        
        # Different prediction: always allow
        if prediction != self.last_prediction:
            return True, "New prediction"
        
        # Same prediction: check cooldown
        if time_since_last >= self.cooldown:
            return True, f"Cooldown elapsed ({time_since_last:.1f}s)"
        
        # Still in cooldown
        remaining = self.cooldown - time_since_last
        return False, f"In cooldown ({remaining:.1f}s remaining)"
    
    def record_prediction(self, prediction: str):
        """Record this prediction as output."""
        self.last_prediction = prediction
        self.last_prediction_time = time.time()
    
    def reset(self):
        """Reset cooldown."""
        self.last_prediction = None
        self.last_prediction_time = 0


class PredictionSmoother:
    """Apply spatial smoothing to confidence scores."""
    
    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: size of confidence smoothing window
        """
        self.window_size = window_size
        self.confidence_window = deque(maxlen=window_size)
    
    def smooth_confidence(self, confidence: float) -> float:
        """Apply smoothing to confidence score."""
        self.confidence_window.append(confidence)
        
        if len(self.confidence_window) < self.window_size:
            return confidence
        
        # Return median confidence
        return np.median(list(self.confidence_window))
    
    def reset(self):
        """Clear confidence window."""
        self.confidence_window.clear()


class GestureStabilityPipeline:
    """
    Complete stability checking pipeline.
    Combines all filters: confidence → stability → smoothing → cooldown.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.90,
        stability_window: int = 15,
        temporal_window: int = 5,
        cooldown_seconds: float = 2.0,
        confidence_smoothing: int = 3
    ):
        """
        Args:
            confidence_threshold: minimum confidence (0-1)
            stability_window: frames needed for stability
            temporal_window: temporal smoothing window
            cooldown_seconds: cooldown time between predictions
            confidence_smoothing: confidence score smoothing window
        """
        self.confidence_validator = ConfidenceValidator(confidence_threshold)
        self.stability_filter = StabilityFilter(stability_window, confidence_threshold)
        self.temporal_smoother = TemporalSmoother(temporal_window)
        self.cooldown_manager = CooldownManager(cooldown_seconds)
        self.confidence_smoother = PredictionSmoother(confidence_smoothing)
        
        self.prediction_history = deque(maxlen=100)
    
    def process_prediction(
        self,
        label: str,
        confidence: float
    ) -> Tuple[bool, Optional[str], str]:
        """
        Process prediction through complete stability pipeline.
        
        Args:
            label: predicted gesture label
            confidence: prediction confidence (0-1)
        
        Returns:
            (should_output, output_label, status_message)
        """
        messages = []
        
        # 1. Smooth confidence
        smoothed_conf = self.confidence_smoother.smooth_confidence(confidence)
        messages.append(f"Raw conf: {confidence:.2%} → Smoothed: {smoothed_conf:.2%}")
        
        # 2. Validate confidence
        conf_valid, conf_msg = self.confidence_validator.is_valid(smoothed_conf)
        messages.append(conf_msg)
        
        if not conf_valid:
            return False, None, " | ".join(messages)
        
        # 3. Add to stability filter
        self.stability_filter.add_prediction(label, smoothed_conf)
        
        # 4. Check stability
        is_stable, stable_label, avg_conf = self.stability_filter.is_stable()
        
        if not is_stable:
            stability_len = len(self.stability_filter.prediction_history)
            messages.append(f"Stability: {stability_len}/15 frames of '{label}'")
            return False, None, " | ".join(messages)
        
        messages.append(f"Stable for 15 frames (avg conf: {avg_conf:.2%})")
        
        # 5. Apply temporal smoothing
        smoothed_label = self.temporal_smoother.smooth(stable_label)
        messages.append(f"Temporal smoothing: {stable_label} → {smoothed_label}")
        
        # 6. Check cooldown
        can_output, cooldown_msg = self.cooldown_manager.can_predict(smoothed_label)
        messages.append(cooldown_msg)
        
        if not can_output:
            return False, None, " | ".join(messages)
        
        # 7. Record prediction
        self.cooldown_manager.record_prediction(smoothed_label)
        self.stability_filter.reset()
        self.temporal_smoother.reset()
        self.prediction_history.append({
            "label": smoothed_label,
            "confidence": avg_conf,
            "timestamp": time.time()
        })
        
        messages.append("✓ PREDICTION CONFIRMED")
        
        return True, smoothed_label, " | ".join(messages)
    
    def get_status(self) -> dict:
        """Get complete pipeline status."""
        return {
            "stability": self.stability_filter.get_status(),
            "recent_predictions": list(self.prediction_history)[-5:],
            "last_prediction": self.prediction_history[-1] if self.prediction_history else None
        }
    
    def reset(self):
        """Reset all filters."""
        self.stability_filter.reset()
        self.temporal_smoother.reset()
        self.cooldown_manager.reset()
        self.confidence_smoother.reset()
        self.prediction_history.clear()


# Preset configurations for different use cases
STABILITY_PRESETS = {
    "conservative": {
        "confidence_threshold": 0.95,
        "stability_window": 20,
        "temporal_window": 5,
        "cooldown_seconds": 3.0,
    },
    "balanced": {
        "confidence_threshold": 0.90,
        "stability_window": 15,
        "temporal_window": 5,
        "cooldown_seconds": 2.0,
    },
    "aggressive": {
        "confidence_threshold": 0.85,
        "stability_window": 10,
        "temporal_window": 3,
        "cooldown_seconds": 1.5,
    },
    "realtime": {
        "confidence_threshold": 0.80,
        "stability_window": 8,
        "temporal_window": 3,
        "cooldown_seconds": 1.0,
    },
}


def create_stability_pipeline(preset: str = "balanced") -> GestureStabilityPipeline:
    """Create a stability pipeline with preset configuration."""
    if preset not in STABILITY_PRESETS:
        preset = "balanced"
    
    config = STABILITY_PRESETS[preset]
    return GestureStabilityPipeline(**config)


if __name__ == "__main__":
    # Example usage
    print("=== Gesture Stability Filter Demo ===\n")
    
    pipeline = create_stability_pipeline("balanced")
    
    # Simulate real predictions
    test_sequence = [
        ("hello", 0.85),
        ("hello", 0.88),
        ("hello", 0.91),
        ("hello", 0.93),
        ("hello", 0.92),
        ("hello", 0.90),
        ("hello", 0.94),
        ("hello", 0.91),
        ("hello", 0.92),
        ("hello", 0.93),
        ("hello", 0.91),
        ("hello", 0.92),
        ("hello", 0.94),
        ("hello", 0.93),
        ("hello", 0.92),  # Should trigger at 15 frames
    ]
    
    for label, conf in test_sequence:
        should_output, output_label, status = pipeline.process_prediction(label, conf)
        
        if should_output:
            print(f"✓ OUTPUT: {output_label} | {status}")
        else:
            print(f"  (filtering) {status}")
    
    print(f"\nFinal Status: {pipeline.get_status()}")
