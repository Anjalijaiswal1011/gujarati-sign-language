# Gesture Recognition Dataset

This folder structure holds the dataset and models for the ASL/Mute Language gesture recognition pipeline.

## Directory Structure
- `raw_images/`: This folder contains the raw image datasets. Each class should have its own folder (e.g., `raw_images/A/`, `raw_images/B/`, etc.). 
- `processed_data/`: Contains CSV files of extracted MediaPipe landmarks. By default, `landmarks.csv` is stored here.
- `models/`: Stores the trained Keras models (`.h5` files) and the `labels.txt` file.

## Getting a Dataset
You have two options to get a dataset:

### Option 1: Download Kaggle ASL Alphabet (Recommended for Alphabets)
1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Download and extract it.
3. Place the class folders inside `raw_images/` so it looks like `raw_images/A/`, `raw_images/B/`, etc.

### Option 2: Custom Webcam Dataset
1. Run `python scripts/data_collection.py` to use your webcam and capture images for custom gestures. 
2. The script will automatically save images into `raw_images/<Class_Name>`.

## Training Pipeline
Once you have images in `raw_images/`:
1. Run `python scripts/preprocess_dataset.py` to convert all images to MediaPipe hand landmarks and save them to `processed_data/landmarks.csv`.
2. Run `python scripts/train_model.py` to train a Neural Network on those landmarks. The output will be saved in the `models/` directory.

The application automatically picks up models from the `dataset/models/` folder.
