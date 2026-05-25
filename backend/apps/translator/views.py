from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from services.enhanced_translation_service import translate_text
from services.nlp_service import NLPProcessor
from services.translation_service import _has_english

class TranslateView(APIView):
    def post(self, request):
        text_to_translate = request.data.get("text")
        source_lang = request.data.get("source_language", "en")
        target_lang = request.data.get("target_language", "gu")
        
        if not text_to_translate:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # We only use GSL NLP corrections if translating from English to Gujarati
            if source_lang == 'en' and target_lang == 'gu':
                words = text_to_translate.strip().split()
                nlp_result = NLPProcessor.correct_sentence(words)
                
                # If NLP correction returns a result with remaining English text, fall back to Google Translate
                if not nlp_result.get("gujarati_translation") or _has_english(nlp_result.get("gujarati_translation")):
                    translation_result = translate_text(text_to_translate, source_lang=source_lang, target_lang=target_lang)
                    gujarati_translation = translation_result.get("translated_text", text_to_translate)
                else:
                    gujarati_translation = nlp_result["gujarati_translation"]
                corrected_text = nlp_result.get("corrected_english", text_to_translate)
            else:
                # Direct neural translation for all other languages
                translation_result = translate_text(text_to_translate, source_lang=source_lang, target_lang=target_lang)
                if translation_result.get("success"):
                    gujarati_translation = translation_result["translated_text"]
                else:
                    gujarati_translation = text_to_translate
                corrected_text = text_to_translate
            
            return Response({
                "original": text_to_translate,
                "corrected": corrected_text,
                "translated": gujarati_translation
            }, status=status.HTTP_200_OK)
        except Exception as e:
            # Fallback: Try enhanced translation directly
            try:
                translation_result = translate_text(text_to_translate, source_lang=source_lang, target_lang=target_lang)
                return Response({
                    "original": text_to_translate,
                    "corrected": text_to_translate,
                    "translated": translation_result.get("translated_text", text_to_translate)
                }, status=status.HTTP_200_OK)
            except:
                return Response({
                    "error": "Translation service temporarily unavailable",
                    "original": text_to_translate
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

