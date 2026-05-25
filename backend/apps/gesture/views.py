import os
import uuid
import cv2
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import GestureHistory
from services.gesture_service import process_gesture_frame
from services.preprocessing import preprocess_for_classifier
from services.constants import ALPHABET_LABELS
from services.training_service import start_training_in_background, get_training_progress

class GestureDetectView(APIView):
    """
    Handles both the AJAX POST request for static detection 
    and the GET request for the live demo page.
    """
    template_name = 'gesture/demo.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        image_data = request.data.get('image_base64')
        mode = request.data.get('mode', 'words')
        if not image_data:
            return Response({"success": False, "message": "No image data provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Process the frame using the ML service pipeline
        result = process_gesture_frame(image_data, mode=mode)
        
        return Response(result, status=status.HTTP_200_OK)

class GestureTrainView(LoginRequiredMixin, TemplateView):
    template_name = 'gesture/train.html'
    login_url = '/login/'

class DatasetStatsAPI(APIView):
    """
    Returns image counts for each class of sign letters and sign words,
    and sequence file counts for dynamic sign gestures.
    """
    def get(self, request):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_images_dir = os.path.join(base_dir, "dataset", "raw_images")
        sequences_dir = os.path.join(base_dir, "dataset", "sequences")
        
        stats = {
            "alphabet": {},
            "words": {},
            "sequences": {},
            "total_images": 0,
            "total_sequences": 0
        }
        
        # Load raw images stats
        if os.path.exists(raw_images_dir):
            alphabet_labels_lower = [l.lower() for l in ALPHABET_LABELS]
            for d in os.listdir(raw_images_dir):
                d_path = os.path.join(raw_images_dir, d)
                if os.path.isdir(d_path):
                    count = len([f for f in os.listdir(d_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    stats["total_images"] += count
                    if d.lower() in alphabet_labels_lower:
                        stats["alphabet"][d] = count
                    else:
                        stats["words"][d] = count

        # Load sequences stats
        if os.path.exists(sequences_dir):
            for d in os.listdir(sequences_dir):
                d_path = os.path.join(sequences_dir, d)
                if os.path.isdir(d_path):
                    count = len([f for f in os.listdir(d_path) if f.endswith('.npy')])
                    stats["total_sequences"] += count
                    stats["sequences"][d] = count
                    
        return Response(stats, status=status.HTTP_200_OK)

class DataCollectionAPI(APIView):
    """
    Saves an image frame captured in the browser to the raw_images/<label> directory.
    Uses hand detector to crop and preprocess before saving.
    """
    def post(self, request):
        image_base64 = request.data.get('image')
        label = request.data.get('label')
        
        if not image_base64 or not label:
            return Response({"success": False, "message": "Image and label are required."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Clean label name to avoid directory traversal
        label = "".join([c for c in label if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not label:
            return Response({"success": False, "message": "Invalid label."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Process and crop hand using existing preprocessing
        processed_image, has_hands, hand_count = preprocess_for_classifier(image_base64)
        if not has_hands or processed_image is None:
            return Response({"success": False, "message": "No hand detected. Keep hand inside the camera guide box."}, status=status.HTTP_400_BAD_REQUEST)
            
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_images_dir = os.path.join(base_dir, "dataset", "raw_images")
        
        # Save image to disk under raw_images/<label>
        label_dir = os.path.join(raw_images_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        filename = f"img_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(label_dir, filename)
        
        cv2.imwrite(filepath, processed_image)
        
        # Count images in this directory
        count = len([f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        return Response({
            "success": True, 
            "message": f"Saved image for class '{label}'",
            "count": count
        }, status=status.HTTP_200_OK)

class StartTrainingAPI(APIView):
    """
    Starts background model training thread.
    """
    def post(self, request):
        mode = request.data.get('mode', 'alphabet') # 'alphabet' or 'words'
        epochs = int(request.data.get('epochs', 5))
        
        started = start_training_in_background(mode, epochs)
        if started:
            return Response({"success": True, "message": f"Background training started for '{mode}' model."}, status=status.HTTP_200_OK)
        else:
            return Response({"success": False, "message": "Training is already in progress. Please wait for it to finish."}, status=status.HTTP_400_BAD_REQUEST)

class TrainingStatusAPI(APIView):
    """
    Returns live training epoch statistics.
    """
    def get(self, request):
        progress = get_training_progress()
        return Response(progress, status=status.HTTP_200_OK)

import shutil

class DatasetFilesAPI(APIView):
    """
    Returns a list of files (images/sequences) inside a specific class directory.
    """
    def get(self, request):
        class_name = request.query_params.get('class_name')
        mode = request.query_params.get('mode') # 'alphabet', 'words', or 'sequences'
        
        if not class_name or not mode:
            return Response({"success": False, "message": "class_name and mode are required."}, status=status.HTTP_400_BAD_REQUEST)
            
        # Clean class_name
        class_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if mode == 'sequences':
            target_dir = os.path.join(base_dir, "dataset", "sequences", class_name)
            exts = ('.npy',)
        else:
            target_dir = os.path.join(base_dir, "dataset", "raw_images", class_name)
            exts = ('.jpg', '.jpeg', '.png')
            
        files_list = []
        if os.path.exists(target_dir):
            for f in sorted(os.listdir(target_dir)):
                if f.lower().endswith(exts):
                    f_path = os.path.join(target_dir, f)
                    files_list.append({
                        "name": f,
                        "size": os.path.getsize(f_path),
                        "created": os.path.getctime(f_path)
                    })
                    
        return Response({
            "success": True,
            "class_name": class_name,
            "mode": mode,
            "files": files_list,
            "count": len(files_list)
        }, status=status.HTTP_200_OK)

class DatasetFileAPI(APIView):
    """
    Serves a single image file as a file response, or returning sequence `.npy` data as JSON.
    """
    def get(self, request):
        class_name = request.query_params.get('class_name')
        mode = request.query_params.get('mode')
        filename = request.query_params.get('filename')
        
        if not class_name or not mode or not filename:
            return Response({"success": False, "message": "class_name, mode, and filename are required."}, status=status.HTTP_400_BAD_REQUEST)
            
        class_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        filename = "".join([c for c in filename if c.isalnum() or c in ('.', '_', '-')]).strip()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if mode == 'sequences':
            file_path = os.path.join(base_dir, "dataset", "sequences", class_name, filename)
        else:
            file_path = os.path.join(base_dir, "dataset", "raw_images", class_name, filename)
            
        if not os.path.exists(file_path):
            return Response({"success": False, "message": "File not found."}, status=status.HTTP_404_NOT_FOUND)
            
        if mode == 'sequences':
            # Load numpy file and return coordinate array as JSON list
            try:
                import numpy as np
                sequence_data = np.load(file_path)
                return Response({
                    "success": True,
                    "filename": filename,
                    "landmarks": sequence_data.tolist()
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"success": False, "message": f"Failed to parse sequence: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Stream image file
            from django.http import HttpResponse
            with open(file_path, "rb") as f:
                return HttpResponse(f.read(), content_type="image/jpeg")

class DeleteDatasetFileAPI(APIView):
    """
    Deletes a specific image or sequence file from the dataset.
    """
    def post(self, request):
        class_name = request.data.get('class_name')
        mode = request.data.get('mode')
        filename = request.data.get('filename')
        
        if not class_name or not mode or not filename:
            return Response({"success": False, "message": "class_name, mode, and filename are required."}, status=status.HTTP_400_BAD_REQUEST)
            
        class_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        filename = "".join([c for c in filename if c.isalnum() or c in ('.', '_', '-')]).strip()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if mode == 'sequences':
            file_path = os.path.join(base_dir, "dataset", "sequences", class_name, filename)
        else:
            file_path = os.path.join(base_dir, "dataset", "raw_images", class_name, filename)
            
        if not os.path.exists(file_path):
            return Response({"success": False, "message": "File not found."}, status=status.HTTP_404_NOT_FOUND)
            
        try:
            os.remove(file_path)
            
            # Clean up class directory if empty
            class_dir = os.path.dirname(file_path)
            if len(os.listdir(class_dir)) == 0:
                os.rmdir(class_dir)
                
            return Response({"success": True, "message": f"Deleted {filename} successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"success": False, "message": f"Failed to delete file: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DeleteDatasetClassAPI(APIView):
    """
    Deletes a whole class directory (sequences or raw images).
    """
    def post(self, request):
        class_name = request.data.get('class_name')
        mode = request.data.get('mode')
        
        if not class_name or not mode:
            return Response({"success": False, "message": "class_name and mode are required."}, status=status.HTTP_400_BAD_REQUEST)
            
        class_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if mode == 'sequences':
            class_dir = os.path.join(base_dir, "dataset", "sequences", class_name)
        else:
            class_dir = os.path.join(base_dir, "dataset", "raw_images", class_name)
            
        if not os.path.exists(class_dir):
            return Response({"success": False, "message": "Class directory not found."}, status=status.HTTP_404_NOT_FOUND)
            
        try:
            shutil.rmtree(class_dir)
            return Response({"success": True, "message": f"Deleted class '{class_name}' dataset successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"success": False, "message": f"Failed to delete class: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelInfoAPI(APIView):
    """
    Returns training information and status of the current classifier models.
    """
    def get(self, request):
        mode = request.query_params.get('mode', 'alphabet') # 'alphabet' or 'words'
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(base_dir, "dataset", "models")
        
        info_file = os.path.join(models_dir, f"{mode}_model_info.json")
        if mode == 'alphabet':
            model_file = os.path.join(models_dir, "alphabet_landmarks_model.h5")
        else:
            model_file = os.path.join(models_dir, "words_lstm_model.h5")
            
        if not os.path.exists(model_file):
            return Response({
                "success": False,
                "trained": False,
                "message": "Model has not been trained yet."
            }, status=status.HTTP_200_OK)
            
        # Try loading metadata info
        import json
        metadata = {}
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
                
        # Get actual model file modified timestamp
        import datetime
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_file)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Merge file details and saved metrics
        response_data = {
            "success": True,
            "trained": True,
            "mode": mode,
            "last_trained": metadata.get("last_trained", modified_time),
            "epochs": metadata.get("epochs", "N/A"),
            "samples": metadata.get("samples", "N/A"),
            "num_classes": metadata.get("num_classes", "N/A"),
            "classes": metadata.get("classes", []),
            "accuracy": metadata.get("accuracy", 0.0),
            "val_accuracy": metadata.get("val_accuracy", 0.0),
            "loss": metadata.get("loss", 0.0),
            "val_loss": metadata.get("val_loss", 0.0)
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
