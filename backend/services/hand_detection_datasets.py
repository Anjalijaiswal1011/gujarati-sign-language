"""
Hand Detection Dataset Integration Guide

This file provides information and utilities for integrating hand detection datasets
for improving gesture recognition accuracy.
"""

import os
from pathlib import Path

# Dataset configuration
DATASET_CONFIGS = {
    'mediapipe': {
        'name': 'MediaPipe Hand Detection',
        'description': 'Pre-trained hand detection model by Google',
        'url': 'https://mediapipe.dev',
        'installation': 'pip install mediapipe',
        'usage': 'Hands detection with 21 hand landmarks'
    },
    'yolo_hand': {
        'name': 'YOLOv8 Hand Detection',
        'description': 'YOLO v8 trained on hand detection dataset',
        'url': 'https://docs.ultralytics.com',
        'installation': 'pip install ultralytics',
        'usage': 'Real-time hand detection and bounding boxes'
    },
    'custom_dataset': {
        'name': 'Custom Hand Dataset',
        'description': 'Your own hand gesture dataset',
        'location': 'services/models/hand_dataset/',
        'format': 'JPG/PNG images with labels'
    }
}

# Popular hand detection datasets
PUBLIC_DATASETS = {
    'Egohands': {
        'url': 'http://vision.soic.indiana.edu/projects/egohands/',
        'samples': '15000+',
        'description': 'First-person hand detection dataset'
    },
    'Cornell Hand': {
        'url': 'http://pr.cs.cornell.edu/grasping/rect_data/data.php',
        'samples': '5000+',
        'description': 'Hand gesture and grasp dataset'
    },
    'Google AI Hand': {
        'url': 'https://mediapipe.dev/solutions/hands',
        'samples': 'Pre-trained',
        'description': 'MediaPipe hand landmarks dataset'
    },
    'Kaggle Hand Gesture': {
        'url': 'https://www.kaggle.com/datasets/gti-upm/leapgestrecog',
        'samples': '10000+',
        'description': 'Static hand gesture recognition dataset'
    }
}

def get_dataset_path():
    """Returns the path to store hand detection datasets"""
    base_path = Path(__file__).resolve().parent
    dataset_path = base_path / 'models' / 'hand_dataset'
    dataset_path.mkdir(parents=True, exist_ok=True)
    return dataset_path

def list_available_datasets():
    """Lists all available hand detection datasets"""
    return DATASET_CONFIGS

def list_public_datasets():
    """Lists publicly available hand detection datasets for download"""
    return PUBLIC_DATASETS

def check_mediapipe_installation():
    """Checks if MediaPipe is installed and accessible"""
    try:
        import mediapipe as mp
        return {'installed': True, 'version': mp.__version__}
    except ImportError:
        return {
            'installed': False,
            'message': 'Install with: pip install mediapipe'
        }

# Installation commands for datasets
INSTALLATION_COMMANDS = {
    'mediapipe': 'pip install mediapipe',
    'ultralytics': 'pip install ultralytics',
    'tensorflow': 'pip install tensorflow',
    'torch': 'pip install torch torchvision',
    'opencv': 'pip install opencv-python'
}

print("""
=== Hand Detection Dataset Integration ===

Available Datasets:
1. MediaPipe (Pre-trained, recommended for quick start)
2. YOLOv8 (Custom training, high accuracy)
3. Custom Dataset (Your own labeled images)

Installation Guide:
""")

for dataset, cmd in INSTALLATION_COMMANDS.items():
    print(f"  {dataset}: {cmd}")

print("\nPublic Datasets to Download:")
for dataset_name, info in PUBLIC_DATASETS.items():
    print(f"  - {dataset_name}: {info['url']}")
