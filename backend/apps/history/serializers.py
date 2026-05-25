from rest_framework import serializers
from .models import TranslationRecord
from apps.favorites.models import FavoritePhrase

class TranslationHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = TranslationRecord
        fields = '__all__'

class FavoritePhraseSerializer(serializers.ModelSerializer):
    class Meta:
        model = FavoritePhrase
        fields = '__all__'
