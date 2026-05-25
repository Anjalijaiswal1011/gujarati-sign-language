from django.contrib import admin
from .models import GestureHistory

@admin.register(GestureHistory)
class GestureHistoryAdmin(admin.ModelAdmin):
    list_display = ('detected_label', 'confidence', 'user', 'created_at')
    list_filter = ('detected_label', 'created_at')
    search_fields = ('detected_label', 'user__username')
