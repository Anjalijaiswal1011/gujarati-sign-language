from django.db import models
from django.contrib.auth.models import User

class GestureHistory(models.Model):
    """
    Stores successful gesture detections triggered by the user via the frontend WebSocket.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='gesture_history', null=True, blank=True)
    detected_label = models.CharField(max_length=50)
    confidence = models.FloatField()
    text_output = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        user_str = self.user.username if self.user else "Guest"
        return f"{user_str} - {self.detected_label} ({self.confidence})"
