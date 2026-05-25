"""
WebSocket Consumer — Gujarati Sign Language Real-Time Detection
==============================================================
Handles:
  1. Alphabet mode  — single-frame MLP prediction
  2. Words mode     — 30-frame sliding-window LSTM prediction
  3. Collect mode   — sequence recording for training

Prediction quality controls:
  - Confidence threshold (60% alphabet / 70% words)
  - Stable-frame verification (5 frames alphabet / 8 frames words)
  - Hand-absence gating (clears buffer instantly)
"""

import json
import base64
import os
import uuid
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from services.gesture_service import process_gesture_frame, process_gesture_sequence
from services.model_loader import get_classifier, LandmarkMLPClassifier, LSTMClassifier
from services.translation_service import translate_to_gujarati
from services.constants import (
    CONFIDENCE_THRESHOLD, WORD_CONFIDENCE_THRESHOLD,
    STABLE_FRAMES_REQUIRED, WORD_STABLE_FRAMES
)
from .models import GestureHistory
from asgiref.sync import sync_to_async


class GestureConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

        # Rolling landmark window for LSTM
        self.landmark_buffer = []
        # Sequence collection buffer
        self.collection_buffer = []

        # Stable-frame tracking (avoid random flickers)
        self.stable_label    = None
        self.stable_count    = 0
        self.last_confirmed  = None   # Last accepted prediction (dedup)
        self.last_word_time  = 0      # ms since last word prediction
        self.absent_count    = 0      # Tracks consecutive hand-absence frames

        # Rolling predictions history for temporal smoothing
        self.raw_label_history = []
        self.raw_conf_history = []

        # Detect which model type is loaded for alphabet
        try:
            from services.model_loader import get_classifier, LandmarkMLPClassifier
            classifier = await sync_to_async(get_classifier)("alphabet")
            model_type = "landmarks" if isinstance(classifier, LandmarkMLPClassifier) else "image"
        except Exception:
            model_type = "image"

        words_ready = await self._check_words_model()
        await self.send(text_data=json.dumps({
            'type': 'init',
            'alphabet_model_type': model_type,
            'words_model_ready': words_ready,
            'message': f'Connected. Alphabet={model_type}'
        }))
        print(f"[Consumer] WebSocket connected. alphabet_model={model_type}")

    async def _check_words_model(self):
        try:
            from services.model_loader import get_classifier, LSTMClassifier
            clf = await sync_to_async(get_classifier)("words")
            return isinstance(clf, LSTMClassifier)
        except Exception:
            return False

    async def disconnect(self, close_code):
        print(f"[Consumer] WebSocket disconnected: {close_code}")

    async def receive(self, text_data):
        try:
            data   = json.loads(text_data)
            mode   = data.get('mode', 'alphabet')
            status = data.get('status', '')

            threshold_val = data.get('threshold')
            try:
                threshold = float(threshold_val) if threshold_val is not None else None
            except (ValueError, TypeError):
                threshold = None

            client_landmarks = data.get('landmarks')
            image_b64        = data.get('image')

            # ── Extract landmarks ──────────────────────────────────────────
            landmarks = None
            if client_landmarks and len(client_landmarks) == 63:
                landmarks = np.array(client_landmarks, dtype=np.float32)
            elif image_b64:
                from services.preprocessing import extract_landmarks_from_b64
                landmarks = await sync_to_async(extract_landmarks_from_b64)(image_b64)

            # ── Early-exit: no-hand / wrong-position frames ────────────────
            if status in ('no_hands', 'incorrect_position') or (
                    landmarks is None and not image_b64):
                self.absent_count += 1
                if self.absent_count >= 5:
                    self.landmark_buffer = []   # Clear LSTM sliding window
                    self.stable_label   = None
                    self.stable_count   = 0
                    self.raw_label_history = []
                    self.raw_conf_history = []
                await self.send(text_data=json.dumps({
                    'type': 'prediction', 'success': False,
                    'status': status or 'no_hands',
                    'message': data.get('message', 'No hands detected')
                }))
                return

            # Reset absent count since hand is present and valid
            self.absent_count = 0

            # ═══════════════════════════════════════════════════════════════
            # COLLECT MODE
            # ═══════════════════════════════════════════════════════════════
            if mode == 'collect':
                label = data.get('label')
                if not label:
                    await self.send(text_data=json.dumps({
                        'type': 'collection_progress', 'success': False,
                        'message': 'Missing label'
                    }))
                    return

                if landmarks is None:
                    await self.send(text_data=json.dumps({
                        'type': 'collection_progress', 'success': False,
                        'message': 'No hand detected — keep hand in the box'
                    }))
                    return

                self.collection_buffer.append(
                    landmarks.tolist() if isinstance(landmarks, np.ndarray) else landmarks
                )
                count = len(self.collection_buffer)

                await self.send(text_data=json.dumps({
                    'type': 'collection_progress', 'success': True,
                    'count': count, 'total': 30
                }))

                if count >= 30:
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    seq_dir  = os.path.join(base_dir, "dataset", "sequences", label)
                    os.makedirs(seq_dir, exist_ok=True)

                    filename = f"seq_{uuid.uuid4().hex[:8]}.npy"
                    filepath = os.path.join(seq_dir, filename)
                    await sync_to_async(np.save)(filepath, np.array(self.collection_buffer))
                    self.collection_buffer = []

                    seq_count = len([f for f in os.listdir(seq_dir) if f.endswith('.npy')])
                    await self.send(text_data=json.dumps({
                        'type': 'collection_success', 'label': label,
                        'message': f"Saved! ({seq_count} total sequences)",
                        'count': seq_count
                    }))
                return

            # ═══════════════════════════════════════════════════════════════
            # ALPHABET MODE — single-frame MLP
            # ═══════════════════════════════════════════════════════════════
            if mode == 'alphabet':
                result = None

                classifier = get_classifier("alphabet")
                if isinstance(classifier, LandmarkMLPClassifier) and landmarks is not None:
                    probs, idx = classifier.getPrediction(landmarks)
                    confidence     = float(np.max(probs)) * 100
                    predicted_label = classifier.labels[idx]

                    # ── Compute Gujarati and prepare result ──────────────────────────────────
                    gujarati_label = translate_to_gujarati(predicted_label)
                    result = {
                        "success": True, "label": predicted_label,
                        "gujarati": gujarati_label, "confidence": confidence,
                        "is_mock": False
                    }
                    
                    # Track stabilization for history saving only
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if predicted_label == self.stable_label:
                            self.stable_count += 1
                        else:
                            self.stable_label = predicted_label
                            self.stable_count = 1
                            
                        if self.stable_count == STABLE_FRAMES_REQUIRED:
                            await self.save_history(result)
                    else:
                        self.stable_label = None
                        self.stable_count = 0

                elif image_b64:
                    result = await sync_to_async(process_gesture_frame)(
                        image_b64, mode="alphabet", threshold=threshold)
                        
                    if result and result.get('success'):
                        await self.save_history(result)

                if result and result.get('success'):
                    lbl = result.get('label')
                    conf = result.get('confidence', 0.0)
                    
                    # Rolling majority vote smoothing (size 5)
                    self.raw_label_history.append(lbl)
                    self.raw_conf_history.append(conf)
                    if len(self.raw_label_history) > 5:
                        self.raw_label_history.pop(0)
                        self.raw_conf_history.pop(0)
                        
                    from collections import Counter
                    label_counts = Counter(self.raw_label_history)
                    smoothed_label = label_counts.most_common(1)[0][0]
                    matching_confs = [c for l, c in zip(self.raw_label_history, self.raw_conf_history) if l == smoothed_label]
                    smoothed_confidence = float(np.mean(matching_confs))
                    
                    smoothed_guj = translate_to_gujarati(smoothed_label)

                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': True,
                        'status': result.get('status', 'success'),
                        'label': smoothed_label,
                        'gujarati': smoothed_guj,
                        'confidence': smoothed_confidence,
                        'text_output': smoothed_label,
                        'mode': mode, 'is_mock': result.get('is_mock', False),
                        'hand_count': 1,
                        'message': result.get('message', '')
                    }))
                elif result:
                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': False,
                        'status': result.get('status', 'low_confidence'),
                        'label': result.get('label', ''),
                        'gujarati': result.get('gujarati', ''),
                        'confidence': result.get('confidence', 0.0),
                        'text_output': result.get('label', ''),
                        'mode': mode, 'is_mock': result.get('is_mock', False),
                        'hand_count': 1,
                        'message': result.get('message', '')
                    }))

            # ═══════════════════════════════════════════════════════════════
            # WORDS MODE — 30-frame sliding-window LSTM
            # ═══════════════════════════════════════════════════════════════
            elif mode == 'words':
                if landmarks is None:
                    # Non-blocking safety check
                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': False,
                        'status': 'no_hands', 'message': 'No hands detected'
                    }))
                    return

                # Append to sliding window (keep last 30 frames)
                self.landmark_buffer.append(
                    landmarks.tolist() if isinstance(landmarks, np.ndarray) else landmarks
                )
                if len(self.landmark_buffer) > 30:
                    self.landmark_buffer.pop(0)

                # Send buffering status until window is full
                if len(self.landmark_buffer) < 30:
                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': False,
                        'status': 'buffering',
                        'message': f'Buffering ({len(self.landmark_buffer)}/30)',
                        'buffer_progress': len(self.landmark_buffer)
                    }))
                    return

                # Run LSTM prediction
                result = await sync_to_async(process_gesture_sequence)(
                    self.landmark_buffer, mode="words", threshold=threshold
                )

                if result and result.get('success'):
                    label      = result['label']
                    confidence = float(result.get('confidence', 0.0))

                    # Rolling majority vote smoothing (size 3)
                    self.raw_label_history.append(label)
                    self.raw_conf_history.append(confidence)
                    if len(self.raw_label_history) > 3:
                        self.raw_label_history.pop(0)
                        self.raw_conf_history.pop(0)
                        
                    from collections import Counter
                    label_counts = Counter(self.raw_label_history)
                    smoothed_label = label_counts.most_common(1)[0][0]
                    matching_confs = [c for l, c in zip(self.raw_label_history, self.raw_conf_history) if l == smoothed_label]
                    smoothed_confidence = float(np.mean(matching_confs))

                    # Confirmed word prediction format
                    gujarati = translate_to_gujarati(smoothed_label)
                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': True,
                        'label': smoothed_label, 'gujarati': gujarati,
                        'confidence': smoothed_confidence,
                        'text_output': smoothed_label, 'mode': mode,
                        'is_mock': result.get('is_mock', False),
                        'hand_count': 1
                    }))
                    
                    # History saving logic
                    word_conf_thresh = threshold if threshold is not None else WORD_CONFIDENCE_THRESHOLD
                    if smoothed_confidence >= word_conf_thresh:
                        if smoothed_label == self.stable_label:
                            self.stable_count += 1
                        else:
                            self.stable_label = smoothed_label
                            self.stable_count = 1
                            
                        if self.stable_count == WORD_STABLE_FRAMES:
                            result['label'] = smoothed_label
                            result['gujarati'] = gujarati
                            result['confidence'] = smoothed_confidence
                            await self.save_history(result)
                            self.stable_count = 0
                    else:
                        self.stable_label = None
                        self.stable_count = 0
                elif result:
                    await self.send(text_data=json.dumps({
                        'type': 'prediction', 'success': False,
                        'status': result.get('status', 'low_confidence'),
                        'label': result.get('label', ''),
                        'gujarati': result.get('gujarati', ''),
                        'confidence': result.get('confidence', 0.0),
                        'text_output': result.get('label', ''),
                        'mode': mode, 'is_mock': result.get('is_mock', False),
                        'hand_count': 1,
                        'message': result.get('message', '')
                    }))

        except Exception as e:
            print(f"[Consumer] Error: {e}")
            import traceback
            traceback.print_exc()
            await self.send(text_data=json.dumps({
                'type': 'prediction', 'success': False,
                'message': f'Server error: {str(e)}'
            }))

    @sync_to_async
    def save_history(self, result):
        # Disabled auto-save for live streaming; history is manually saved via confirm button.
        pass
