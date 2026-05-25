# SignTranslate | Accessibility Web App

## 🎯 Purpose
A modern, simple, and highly accessible web application designed to break communication barriers for deaf, mute, and differently-abled users. It acts as an interactive assistant using standard web APIs, Django machine learning integrations, and soft UI styling.

## ✨ Features
- **English to Gujarati Translation**: Instantly translates English text or recognized voice into readable Gujarati.
- **Voice Input**: Uses the native browser SpeechRecognition API to avoid typing.
- **Sign Language Gesture Detection**: 
  - Captures video frames using `navigator.mediaDevices`.
  - Extracts 63 hand landmarks using **MediaPipe Hands**.
  - Processes extracted features via a classification model to return labels like *Hello, Yes, No, Help*.
- **History & Favorites**: Save essential translations or interaction moments securely inside your logged-in profile.
- **Accessible Soft UI**: Developed with pastel blues, lavenders, and mint green to reduce eye strain, while keeping text contrast high.

## 📁 Clean Structure (Apps & Services)
- `apps/`: Houses all Django apps (`users`, `history`, `gesture`, `translator`, `favorites`).
- `services/`: Houses all heavy lifting to keep Django views thin:
  - `preprocessing.py` (handles image decoding and MediaPipe landmarks).
  - `model_loader.py` (smoothly handles ML loading and graceful placeholder fallback).
  - `gesture_service.py` (the entire pipeline wrapped into one reusable function).
- `frontend/`: Standard Django templates and static styling decoupled from the backend apps.

## 🚀 Setup & Execution

1. **Activate the Virtual Environment**:
   ```powershell
   .\venv\Scripts\activate.ps1
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Migrations** (if changing models):
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
4. **Start the Server**:
   ```bash
   python manage.py runserver
   ```
5. Open your browser continuously to `http://127.0.0.1:8000/`.

## 🧠 Replacing the ML Model
For your Viva or final presentation, point out `services/model_loader.py`. Right now, it perfectly simulates an ML model structure. To add a real model:
1. Export a `.pkl` from scikit-learn or similar.
2. Put the file inside `services/models/gesture_model.pkl`.
3. The script automatically detects the real file and begins sending real predictions to the frontend! No extra UI code needed.
