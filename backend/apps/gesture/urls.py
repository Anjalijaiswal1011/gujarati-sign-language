from django.urls import path
from .views import (
    GestureDetectView, GestureTrainView,
    DatasetStatsAPI, DataCollectionAPI,
    StartTrainingAPI, TrainingStatusAPI,
    DatasetFilesAPI, DatasetFileAPI,
    DeleteDatasetFileAPI, DeleteDatasetClassAPI,
    ModelInfoAPI
)

# We remove app_name to allow flat reversal from core templates (e.g. {% url 'gesture' %})
# This ensures student project templates remain simple.

urlpatterns = [
    path('detect/', GestureDetectView.as_view(), name='gesture-detect'),
    path('', GestureDetectView.as_view(template_name='gesture/demo.html'), name='gesture'),
    path('train/', GestureTrainView.as_view(), name='gesture-train'),
    path('api/dataset/stats/', DatasetStatsAPI.as_view(), name='api-dataset-stats'),
    path('api/dataset/collect/', DataCollectionAPI.as_view(), name='api-dataset-collect'),
    path('api/model/train/', StartTrainingAPI.as_view(), name='api-model-train'),
    path('api/model/train/status/', TrainingStatusAPI.as_view(), name='api-model-train-status'),
    
    # Dataset and model management APIs
    path('api/dataset/files/', DatasetFilesAPI.as_view(), name='api-dataset-files'),
    path('api/dataset/file/', DatasetFileAPI.as_view(), name='api-dataset-file'),
    path('api/dataset/delete-file/', DeleteDatasetFileAPI.as_view(), name='api-dataset-delete-file'),
    path('api/dataset/delete-class/', DeleteDatasetClassAPI.as_view(), name='api-dataset-delete-class'),
    path('api/model/info/', ModelInfoAPI.as_view(), name='api-model-info'),
]
