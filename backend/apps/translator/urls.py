from django.urls import path
from .views import TranslateView
from .voice_views import (
    VoiceConverterView,
    SpeechToTextView,
    EnhancedTranslationView,
    TranslateBatchView,
    SupportedLanguagesView,
    LanguageDetectionView
)

urlpatterns = [
    # Original translation endpoint
    path('translate/', TranslateView.as_view(), name='translate'),
    
    # Enhanced translation endpoints
    path('enhanced/translate/', EnhancedTranslationView.as_view(), name='enhanced-translate'),
    path('batch/', TranslateBatchView.as_view(), name='batch-translate'),
    path('languages/', SupportedLanguagesView.as_view(), name='supported-languages'),
    path('detect/', LanguageDetectionView.as_view(), name='detect-language'),
    
    # Voice conversion endpoints (Gujarati and multi-language support)
    path('voice/text-to-speech/', VoiceConverterView.as_view(), name='text-to-speech'),
    path('voice/speech-to-text/', SpeechToTextView.as_view(), name='speech-to-text'),
]
