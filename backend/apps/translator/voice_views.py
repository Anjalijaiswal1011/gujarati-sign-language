"""
API endpoints for voice conversion and enhanced translation services
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from services.voice_converter import text_to_speech, speech_to_text
from services.enhanced_translation_service import (
    translate_text,
    translate_batch,
    get_supported_languages,
    detect_language
)
from django.core.files.storage import default_storage
from django.conf import settings
import os

class VoiceConverterView(APIView):
    """
    API endpoint for text-to-speech conversion in multiple languages including Gujarati
    
    POST /api/voice/text-to-speech/
    Request body:
    {
        "text": "Hello world",
        "language": "gu"  (optional, default: "gu" for Gujarati)
    }
    """
    def post(self, request):
        text = request.data.get('text')
        language = request.data.get('language', 'gu')  # Default to Gujarati
        
        if not text:
            return Response(
                {'error': 'Text parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = text_to_speech(text, language, output_file=None)
        
        if result['success']:
            import base64
            try:
                audio_b64 = base64.b64encode(result['audio_bytes']).decode('utf-8')
                return Response({
                    'success': True,
                    'message': 'Speech generated successfully',
                    'language': language,
                    'audio_base64': audio_b64
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'success': False,
                    'error': f'Failed to encode audio: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({
                'success': False,
                'error': result.get('error', 'Speech generation failed')
            }, status=status.HTTP_400_BAD_REQUEST)

class SpeechToTextView(APIView):
    """
    API endpoint for speech-to-text conversion
    
    POST /api/voice/speech-to-text/
    Request: Upload audio file
    """
    def post(self, request):
        audio_file = request.FILES.get('audio')
        
        if not audio_file:
            return Response(
                {'error': 'Audio file is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save temporary file
        temp_path = default_storage.save(f'temp_{audio_file.name}', audio_file)
        
        try:
            result = speech_to_text(temp_path)
            
            # Clean up temporary file
            default_storage.delete(temp_path)
            
            if result['success']:
                return Response({
                    'success': True,
                    'text': result['text']
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'success': False,
                    'error': result.get('error', 'Speech recognition failed')
                }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            if default_storage.exists(temp_path):
                default_storage.delete(temp_path)
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class EnhancedTranslationView(APIView):
    """
    Enhanced translation endpoint supporting multiple languages including Gujarati
    
    POST /api/translation/translate/
    Request body:
    {
        "text": "Hello world",
        "source_language": "en",
        "target_language": "gu"
    }
    """
    def post(self, request):
        text = request.data.get('text')
        source_lang = request.data.get('source_language', 'en')
        target_lang = request.data.get('target_language', 'gu')
        
        if not text:
            return Response(
                {'error': 'Text parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = translate_text(text, source_lang, target_lang)
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

class TranslateBatchView(APIView):
    """
    Batch translation endpoint for translating multiple texts at once
    
    POST /api/translation/batch/
    Request body:
    {
        "texts": ["Hello", "Good morning"],
        "source_language": "en",
        "target_language": "gu"
    }
    """
    def post(self, request):
        texts = request.data.get('texts')
        source_lang = request.data.get('source_language', 'en')
        target_lang = request.data.get('target_language', 'gu')
        
        if not texts or not isinstance(texts, list):
            return Response(
                {'error': 'Texts parameter must be a list'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = translate_batch(texts, source_lang, target_lang)
        
        return Response({
            'success': True,
            'translations': results,
            'count': len(results)
        }, status=status.HTTP_200_OK)

class SupportedLanguagesView(APIView):
    """
    Get list of supported languages
    
    GET /api/translation/languages/
    """
    def get(self, request):
        languages = get_supported_languages()
        return Response({
            'success': True,
            'languages': languages,
            'total': len(languages)
        }, status=status.HTTP_200_OK)

class LanguageDetectionView(APIView):
    """
    Detect the language of input text
    
    POST /api/translation/detect/
    Request body:
    {
        "text": "Namaste"
    }
    """
    def post(self, request):
        text = request.data.get('text')
        
        if not text:
            return Response(
                {'error': 'Text parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = detect_language(text)
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
