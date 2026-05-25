from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import TranslationRecord
from apps.favorites.models import FavoritePhrase
from .serializers import TranslationHistorySerializer, FavoritePhraseSerializer

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

class HistoryListView(APIView):
    def get(self, request):
        if request.user.is_authenticated:
            histories = TranslationRecord.objects.filter(user=request.user)
        else:
            histories = TranslationRecord.objects.filter(user__isnull=True)
        serializer = TranslationHistorySerializer(histories, many=True)
        return Response(serializer.data)

    def post(self, request):
        user_id = request.user.id if request.user.is_authenticated else None
        
        data = request.data.copy()
        data['user'] = user_id
        
        # Extract base64 image if present
        image_b64 = data.get('image_base64')
        if 'image_base64' in data:
            del data['image_base64']
            
        serializer = TranslationHistorySerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            
            # Save decoded image to FileField if provided
            if image_b64:
                import uuid
                import base64
                from django.core.files.base import ContentFile
                try:
                    if ';base64,' in image_b64:
                        format_str, imgstr = image_b64.split(';base64,')
                        ext = format_str.split('/')[-1]
                    else:
                        imgstr = image_b64
                        ext = 'jpg'
                    
                    filename = f"gesture_{uuid.uuid4().hex[:8]}.{ext}"
                    image_data = ContentFile(base64.b64decode(imgstr), name=filename)
                    instance.detected_image.save(filename, image_data, save=True)
                except Exception as e:
                    print(f"[HistoryView] Error decoding base64 image: {e}")
            
            # --- WEB-SOCKET INTEGRATION ---
            channel_layer = get_channel_layer()
            image_url = instance.detected_image.url if instance.detected_image else ""
            async_to_sync(channel_layer.group_send)(
                "global_updates",
                {
                    "type": "app_notification",
                    "data": {
                        "english_text": instance.english_text,
                        "gujarati_text": instance.gujarati_text,
                        "source": instance.source_type,
                        "confidence": instance.confidence,
                        "detected_image": image_url,
                        "timestamp": instance.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            )
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class HistoryDeleteView(APIView):
    def delete(self, request, pk):
        if request.user.is_authenticated:
            history = get_object_or_404(TranslationRecord, pk=pk, user=request.user)
        else:
            history = get_object_or_404(TranslationRecord, pk=pk, user__isnull=True)
        history.delete()
        return Response({"message": "History deleted"}, status=status.HTTP_204_NO_CONTENT)


class FavoriteListView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
        favorites = FavoritePhrase.objects.filter(user=request.user)
        serializer = FavoritePhraseSerializer(favorites, many=True)
        return Response(serializer.data)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
            
        data = request.data.copy()
        data['user'] = request.user.id
        
        serializer = FavoritePhraseSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
