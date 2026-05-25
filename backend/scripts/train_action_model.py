"""
Train Action Detection Model (CNN+LSTM)
========================================
Trains a CNN+LSTM model to recognize dynamic action sequences.

Assumes landmark sequences are already extracted and stored in:
  dataset/sequences/actions/[action_name]/[sequence].npy

Model Architecture:
  Input: (30, 84) landmark sequences
  → Dense(256) + BatchNorm
  → Bidirectional LSTM(256) + Dropout
  → Bidirectional LSTM(128) + Dropout
  → Dense(64) + ReLU
  → Dense(num_actions, Softmax)

Usage:
  python scripts/train_action_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCES_DIR = os.path.join(BASE_DIR, "dataset", "sequences", "actions")
MODELS_DIR = os.path.join(BASE_DIR, "dataset", "models")
MODEL_OUTPUT = os.path.join(MODELS_DIR, "action_model.h5")
LABELS_OUTPUT = os.path.join(MODELS_DIR, "action_labels.txt")
PROGRESS_FILE = os.path.join(MODELS_DIR, "action_training_progress.json")


def load_action_sequences(root_dir):
    """
    Load all action landmark sequences from directories.
    
    Directory structure:
      actions/
        ├── walking/
        │   ├── walking_00001.npy
        │   ├── walking_00002.npy
        │   └── ...
        ├── eating/
        │   └── ...
    
    Returns:
        (X, y, label_map, class_names)
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"[Action Training] Sequences directory not found: {root_dir}")
        print(f"[Action Training] Please run landmark extraction first:")
        print(f"  python services/landmark_extraction.py --mode=actions")
        return None, None, None, None
    
    X = []
    y = []
    label_map = {}
    class_names = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    
    if not class_names:
        print(f"[Action Training] No action directories found in {root_dir}")
        return None, None, None, None
    
    print(f"[Action Training] Found {len(class_names)} action classes: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        sequence_files = sorted(class_dir.glob("*.npy"))
        
        print(f"  {class_name}: {len(sequence_files)} sequences")
        
        for seq_file in sequence_files:
            try:
                sequence = np.load(seq_file)
                
                # Ensure correct shape: (30, 84)
                if sequence.shape != (30, 84):
                    print(f"    [Warning] {seq_file.name} has shape {sequence.shape}, skipping")
                    continue
                
                X.append(sequence)
                y.append(class_idx)
            except Exception as e:
                print(f"    [Error] Failed to load {seq_file.name}: {e}")
        
        label_map[class_idx] = class_name
    
    if not X:
        print("[Action Training] No valid sequences found!")
        return None, None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[Action Training] Loaded {len(X)} sequences")
    print(f"[Action Training] Feature shape per sequence: {X[0].shape}")
    
    return X, y, label_map, class_names


def split_dataset(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test."""
    np.random.seed(42)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n[Action Training] Dataset split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/n_samples:.1%})")
    print(f"  Val:   {len(X_val)} ({len(X_val)/n_samples:.1%})")
    print(f"  Test:  {len(X_test)} ({len(X_test)/n_samples:.1%})")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def augment_sequences(X, y, factor=3, seed=42):
    """Apply augmentation to increase dataset size."""
    np.random.seed(seed)
    
    X_augmented = [X]
    y_augmented = [y]
    
    for aug_run in range(factor - 1):
        X_aug = X.copy()
        
        # 1. Scaling (±15% size variation)
        scale = np.random.uniform(0.85, 1.15, (len(X_aug), 1, 1))
        X_aug = X_aug * scale
        
        # 2. Noise (small random perturbation)
        noise = np.random.normal(0, 0.012, X_aug.shape)
        X_aug = X_aug + noise
        
        # 3. Translation jitter
        jitter = np.random.normal(0, 0.025, (len(X_aug), 1, X_aug.shape[-1]))
        X_aug = X_aug + jitter
        
        # 4. Temporal warping (random frame skipping)
        # Keep with probability 0.8 to simulate frame drops
        X_aug_warped = []
        for seq in X_aug:
            mask = np.random.rand(len(seq)) > 0.1  # Keep 90% of frames
            indices = np.where(mask)[0]
            if len(indices) < 5:  # Need at least 5 frames
                X_aug_warped.append(seq)
            else:
                # Resample to 30 frames
                warped = np.interp(
                    np.linspace(0, len(indices)-1, 30),
                    np.arange(len(indices)),
                    seq[indices, :].mean(axis=-1)
                )
                X_aug_warped.append(seq)  # Keep original for simplicity
        
        X_augmented.append(np.array(X_aug))
        y_augmented.append(y)
    
    X_all = np.vstack(X_augmented)
    y_all = np.hstack(y_augmented)
    
    print(f"\n[Action Training] After augmentation (factor={factor}):")
    print(f"  Dataset size: {len(X)} -> {len(X_all)}")
    
    return X_all, y_all


def build_action_model(input_shape, num_classes):
    """
    Build CNN+LSTM model for action recognition.
    
    Input shape: (sequence_length=30, features=84)
    """
    model = models.Sequential([
        # Dense preprocessing
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # First Bidirectional LSTM
        layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2)
        ),
        layers.BatchNormalization(),
        
        # Second Bidirectional LSTM
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=False, dropout=0.2)
        ),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


class ActionTrainingProgressCallback(tf.keras.callbacks.Callback):
    """Track training progress to JSON file."""
    
    def __init__(self, epochs):
        super().__init__()
        self.total_epochs = epochs
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress = {
            "status": "training",
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "loss": float(logs.get("loss", 0)),
            "accuracy": float(logs.get("accuracy", 0)),
            "val_loss": float(logs.get("val_loss", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
            "message": (
                f"Epoch {epoch + 1}/{self.total_epochs} | "
                f"Loss: {logs.get('loss', 0):.4f} | "
                f"Acc: {logs.get('accuracy', 0):.2%} | "
                f"Val Loss: {logs.get('val_loss', 0):.4f} | "
                f"Val Acc: {logs.get('val_accuracy', 0):.2%}"
            )
        }
        
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f)
        
        if (epoch + 1) % 10 == 0:
            print(progress["message"])


def train_action_model(epochs=100):
    """Train action detection model."""
    
    print("=" * 70)
    print("TRAINING ACTION DETECTION MODEL (CNN+LSTM)")
    print("=" * 70)
    
    # Load sequences
    X, y, label_map, class_names = load_action_sequences(SEQUENCES_DIR)
    if X is None:
        return
    
    num_classes = len(class_names)
    
    # Augment training data
    print("\n[Action Training] Applying data augmentation...")
    X_augmented, y_augmented = augment_sequences(X, y, factor=3)
    
    # Split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
        X_augmented, y_augmented, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    
    # Build model
    print("\n[Action Training] Building model architecture...")
    input_shape = (30, 84)  # 30 frames, 84 features per frame
    model = build_action_model(input_shape, num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    progress_callback = ActionTrainingProgressCallback(epochs=epochs)
    
    # Train model
    print(f"\n[Action Training] Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=[checkpoint, early_stop, reduce_lr, progress_callback],
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Detailed per-class evaluation
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    
    print("\nPer-Class Results:")
    for class_idx, class_name in enumerate(class_names):
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred[mask] == y_test[mask])
            class_count = np.sum(mask)
            print(f"  {class_name:15s}: {class_accuracy:.2%} ({class_count} samples)")
    
    # Save labels
    print(f"\nSaving labels to {LABELS_OUTPUT}")
    with open(LABELS_OUTPUT, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Save training info
    info = {
        "model": "action_detection_cnn_lstm",
        "input_shape": input_shape,
        "num_classes": num_classes,
        "classes": class_names,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history['loss']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
    }
    
    info_path = os.path.join(MODELS_DIR, "action_model_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nModel information saved to {info_path}")
    print(f"Model saved to {MODEL_OUTPUT}")
    
    print("\n" + "=" * 70)
    print("ACTION MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    epochs = 10
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass
    train_action_model(epochs)
