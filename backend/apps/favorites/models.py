from django.db import models
from django.contrib.auth.models import User

class FavoritePhrase(models.Model):
    """
    Stores users' favorite or commonly used translated phrases for rapid access.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='favorites')
    title = models.CharField(max_length=100, blank=True, help_text="Optional short label")
    english_text = models.CharField(max_length=255)
    gujarati_text = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.english_text}"
