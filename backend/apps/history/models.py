from django.db import models
from django.contrib.auth.models import User

class TranslationRecord(models.Model):
    """
    Stores translation history entries, supporting users and guests.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='translations', null=True, blank=True)
    english_text = models.TextField()
    gujarati_text = models.TextField()
    source_type = models.CharField(max_length=20, default='Text')
    confidence = models.FloatField(default=1.0)
    detected_image = models.FileField(upload_to='detected_gestures/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        user_str = self.user.username if self.user else "Guest"
        return f"{user_str} - {self.english_text[:20]} ({self.confidence})"

