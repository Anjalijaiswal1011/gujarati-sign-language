"""
Enhanced Translation Service with Gujarati Language Support
Includes text-to-speech and voice conversion capabilities
"""

from deep_translator import GoogleTranslator

# Supported languages with codes
SUPPORTED_LANGUAGES = {
    'english': 'en',
    'gujarati': 'gu',
    'hindi': 'hi',
    'marathi': 'mr',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'chinese': 'zh-CN',
    'japanese': 'ja',
    'arabic': 'ar',
}

def translate_text(text, source_lang='en', target_lang='gu'):
    """
    Translates text between supported languages.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code (default: 'en')
        target_lang (str): Target language code (default: 'gu' for Gujarati)
        
    Returns:
        dict: Translation result with success flag and translated text
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return {
            'success': True,
            'original_text': text,
            'translated_text': translated,
            'source_language': source_lang,
            'target_language': target_lang
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Translation error: {str(e)}',
            'message': 'Failed to translate text. Please try again.'
        }

def translate_to_gujarati(english_text):
    """
    Translates English text to Gujarati.
    Backward compatible function.
    
    Args:
        english_text (str): English text to translate
        
    Returns:
        str: Gujarati translated text or error message
    """
    result = translate_text(english_text, source_lang='en', target_lang='gu')
    if result['success']:
        return result['translated_text']
    return result['message']

def translate_batch(texts, source_lang='en', target_lang='gu'):
    """
    Translates multiple texts at once.
    
    Args:
        texts (list): List of text strings to translate
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        list: List of translation results
    """
    results = []
    for text in texts:
        result = translate_text(text, source_lang, target_lang)
        results.append(result)
    return results

def get_supported_languages():
    """Returns list of supported languages"""
    return SUPPORTED_LANGUAGES

def detect_language(text):
    """
    Detects the language of input text.
    
    Args:
        text (str): Text to detect language for
        
    Returns:
        dict: Detected language and confidence
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        detected_lang = blob.detect_language()
        return {
            'success': True,
            'detected_language': detected_lang,
            'language_name': SUPPORTED_LANGUAGES.get(detected_lang, 'Unknown')
        }
    except ImportError:
        return {
            'success': False,
            'error': 'TextBlob not installed. Run: pip install textblob',
            'message': 'Language detection not available'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
