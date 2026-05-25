"""
Real-Time Integration Guide & Implementation
==============================================
Complete pipeline for real-time Gujarati Sign Language detection and translation.

Architecture:
  Webcam → MediaPipe Landmarks → Sequence Buffer → 
  Action/Word LSTM → Stability Filter → Translation → Output

This file provides:
1. Integration example
2. WebSocket consumer implementation
3. Real-time pipeline orchestrator
"""

import asyncio
import numpy as np
import cv2
from typing import Optional, Dict, Any

from services.landmark_extraction import initialize_landmarks_detector, normalize_landmarks
from services.action_detection_service import ActionDetectionService, SequenceBuffer
from services.gesture_stability_filter import create_stability_pipeline
from services.gesture_service import process_gesture_frame
from services.constants import GUJARATI_WORD_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Detection Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeGestureOrchestrator:
    """
    Main orchestrator for real-time gesture detection.
    
    Handles:
    - Frame capture and preprocessing
    - Landmark extraction
    - Static gesture (alphabet) detection
    - Dynamic action detection
    - Stability filtering
    - Gujarati translation
    - WebSocket output
    """
    
    def __init__(self, mode: str = "balanced"):
        """
        Initialize orchestrator.
        
        Args:
            mode: stability filter preset ("aggressive", "balanced", "conservative")
        """
        self.mode = mode
        
        # Components
        self.mp_detector = initialize_landmarks_detector()
        self.action_service = ActionDetectionService(mode=mode)
        self.stability_pipeline = create_stability_pipeline(mode)
        
        # Buffers
        self.sequence_buffer = SequenceBuffer(max_length=30)
        
        # Mode: "alphabet" or "words" or "actions"
        self.detection_mode = "words"  # Default to word detection
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detections = []
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single video frame.
        
        Args:
            frame: OpenCV BGR image frame
        
        Returns:
            Detection result dictionary
        """
        self.frame_count += 1
        
        # Try to extract landmarks
        landmarks = self._extract_landmarks(frame)
        
        if landmarks is None:
            return {
                "success": False,
                "status": "no_hands",
                "message": "No hands detected",
                "hands_detected": False,
                "frame_number": self.frame_count
            }
        
        # Route based on detection mode
        if self.detection_mode == "alphabet":
            return self._process_alphabet_frame(frame)
        elif self.detection_mode == "words":
            return self._process_word_sequence(landmarks)
        elif self.detection_mode == "actions":
            return self._process_action_sequence(landmarks)
        else:
            return {
                "success": False,
                "status": "invalid_mode",
                "message": f"Invalid detection mode: {self.detection_mode}",
                "frame_number": self.frame_count
            }
    
    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract and normalize landmarks from frame."""
        try:
            if self.mp_detector is None:
                return None
            
            import mediapipe
            from mediapipe.tasks.python import vision
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = vision.Image(
                image_format=vision.ImageFormat.SRGB,
                data=frame_rgb
            )
            
            result = self.mp_detector.detect(mp_image)
            
            if not result.hand_landmarks:
                return None
            
            # Get first hand
            hand_landmarks = result.hand_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in hand_landmarks]
            
            h, w = frame.shape[:2]
            normalized = normalize_landmarks(landmarks, w, h)
            
            return normalized
        
        except Exception as e:
            print(f"[Orchestrator] Error extracting landmarks: {e}")
            return None
    
    def _process_alphabet_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process single frame for alphabet detection.
        Uses image-based CNN classifier.
        """
        import base64
        
        try:
            # Convert to base64 for gesture_service
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Use existing gesture service
            result = process_gesture_frame(frame_b64, mode="alphabet")
            
            if result.get("success"):
                self.detection_count += 1
                self.last_detections.append(result.get("label"))
            
            return {
                "success": result.get("success", False),
                "status": result.get("status", "unknown"),
                "label": result.get("label"),
                "confidence": result.get("confidence", 0),
                "gujarati": self._translate_label(result.get("label")),
                "frame_number": self.frame_count
            }
        
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "message": str(e),
                "frame_number": self.frame_count
            }
    
    def _process_word_sequence(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Process 30-frame sequence for word detection.
        Uses LSTM on landmark sequences.
        """
        # Add to buffer
        is_full = self.sequence_buffer.add(landmarks)
        
        if not is_full:
            return {
                "success": False,
                "status": "buffering",
                "message": f"Buffering... {self.sequence_buffer.size()}/30 frames",
                "progress": self.sequence_buffer.progress(),
                "frame_number": self.frame_count
            }
        
        # Get sequence and run through action service
        # (which has LSTM model)
        sequence = self.sequence_buffer.get_sequence()
        result = self.action_service.detect_action(sequence)
        
        if result["success"]:
            self.detection_count += 1
            self.last_detections.append(result["action"])
            self.sequence_buffer.clear()  # Reset after detection
        
        return {
            "success": result["success"],
            "status": result["status"],
            "label": result.get("action"),
            "confidence": result.get("confidence", 0),
            "gujarati": result.get("gujarati"),
            "is_stable": result.get("is_stable", False),
            "buffer_progress": self.sequence_buffer.progress(),
            "frame_number": self.frame_count
        }
    
    def _process_action_sequence(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Process video sequence for action detection.
        Uses CNN+LSTM on landmark sequences.
        """
        return self._process_word_sequence(landmarks)  # Same as word detection
    
    def _translate_label(self, label: str) -> Optional[str]:
        """Translate label to Gujarati."""
        if not label:
            return None
        
        # Try direct mapping
        if label in GUJARATI_WORD_MAP:
            return GUJARATI_WORD_MAP[label]
        
        # Try lowercase
        label_lower = label.lower()
        if label_lower in GUJARATI_WORD_MAP:
            return GUJARATI_WORD_MAP[label_lower]
        
        # Try with underscore
        label_underscore = label.replace(" ", "_").lower()
        if label_underscore in GUJARATI_WORD_MAP:
            return GUJARATI_WORD_MAP[label_underscore]
        
        return None
    
    def set_detection_mode(self, mode: str):
        """Set detection mode: alphabet, words, or actions."""
        if mode in ["alphabet", "words", "actions"]:
            self.detection_mode = mode
            self.reset()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def reset(self):
        """Reset buffers and filters."""
        self.sequence_buffer.clear()
        self.action_service.reset_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "mode": self.detection_mode,
            "total_frames": self.frame_count,
            "total_detections": self.detection_count,
            "recent_detections": self.last_detections[-10:],
            "buffer_status": self.sequence_buffer.progress(),
            "mp_detector_ready": self.mp_detector is not None
        }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Consumer for Django Channels
# ─────────────────────────────────────────────────────────────────────────────

class GestureWebSocketConsumer:
    """
    Django Channels WebSocket consumer for real-time gesture detection.
    
    Integrates with the existing gesture detection system.
    
    Usage in Django Channels:
        from channels.generic.websocket import AsyncWebsocketConsumer
        
        class GestureConsumer(AsyncWebsocketConsumer):
            async def connect(self):
                self.orchestrator = RealTimeGestureOrchestrator(mode="balanced")
                
            async def receive(self, text_data):
                result = self.process_frame_data(text_data)
                await self.send(json.dumps(result))
    """
    
    def __init__(self):
        """Initialize WebSocket consumer."""
        self.orchestrator = RealTimeGestureOrchestrator(mode="balanced")
    
    def process_frame_data(self, frame_b64: str) -> Dict[str, Any]:
        """
        Process base64-encoded frame from frontend.
        
        Args:
            frame_b64: base64-encoded frame
        
        Returns:
            Detection result
        """
        try:
            import base64
            
            # Decode base64 to OpenCV image
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',')[1]
            
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"success": False, "status": "decode_error"}
            
            # Process frame
            result = self.orchestrator.process_frame(frame)
            
            return result
        
        except Exception as e:
            return {
                "success": False,
                "status": "error",
                "message": str(e)
            }


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Webcam Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam_demo(mode: str = "balanced"):
    """
    Run real-time webcam demo.
    
    Args:
        mode: detection mode ("alphabet", "words", or "actions")
    """
    print("=" * 70)
    print("REAL-TIME GUJARATI SIGN LANGUAGE DETECTION")
    print("=" * 70)
    
    orchestrator = RealTimeGestureOrchestrator(mode="balanced")
    orchestrator.set_detection_mode(mode)
    
    print(f"\nDetection Mode: {mode.upper()}")
    print("Controls: q = quit | a = alphabet | w = words | c = actions")
    print("-" * 70)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        result = orchestrator.process_frame(frame)
        
        # Display information
        h, w = frame.shape[:2]
        
        # Status text
        if result["success"]:
            status_color = (0, 255, 0)  # Green
            status_text = f"✓ {result['label']} ({result['confidence']:.0%})"
            gujarati_text = f"Gujarati: {result['gujarati']}"
        else:
            status_color = (0, 165, 255)  # Orange
            status_text = f"○ {result['message']}"
            gujarati_text = ""
        
        # Draw on frame
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, status_color, 2)
        
        if gujarati_text:
            cv2.putText(frame, gujarati_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
        
        # Mode and stats
        mode_text = f"Mode: {orchestrator.detection_mode.upper()}"
        stats_text = f"Frames: {orchestrator.frame_count} | Detected: {orchestrator.detection_count}"
        
        cv2.putText(frame, mode_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        cv2.putText(frame, stats_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        
        # Progress bar for sequence buffering
        if "buffer_progress" in result:
            bar_width = 300
            bar_height = 20
            bar_x = w - bar_width - 10
            bar_y = 10
            
            fill_width = int(bar_width * result["buffer_progress"])
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), 1)
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                             (0, 255, 0), -1)
            
            progress_text = f"{result['buffer_progress']:.0%}"
            cv2.putText(frame, progress_text, (bar_x + 5, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Gujarati Sign Language Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            orchestrator.set_detection_mode("alphabet")
        elif key == ord('w'):
            orchestrator.set_detection_mode("words")
        elif key == ord('c'):
            orchestrator.set_detection_mode("actions")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total Frames Processed: {orchestrator.frame_count}")
    print(f"Total Detections: {orchestrator.detection_count}")
    print(f"Detection Rate: {orchestrator.detection_count / max(1, orchestrator.frame_count):.2%}")
    print(f"Recent Detections: {orchestrator.last_detections[-10:]}")


if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "words"
    
    try:
        run_webcam_demo(mode)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
