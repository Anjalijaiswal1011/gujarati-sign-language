"""
Voice/Text-to-Speech Converter Service
Converts text to speech in multiple languages including Gujarati
"""
try:
    from gtts import gTTS
    import os
except ImportError:
    gTTS = None

def text_to_speech(text, language='gu', output_file='speech.mp3'):
    """
    Converts text to speech in specified language.
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code ('gu' for Gujarati, 'en' for English)
        output_file (str): Output file path for MP3, or None to return bytes
        
    Returns:
        dict: Success flag and file path/audio bytes or error message
    """
    if not gTTS:
        return {
            'success': False,
            'error': 'gTTS library not installed. Run: pip install gtts'
        }
    
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        if output_file is None:
            from io import BytesIO
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return {
                'success': True,
                'audio_bytes': fp.read()
            }
        else:
            tts.save(output_file)
            return {
                'success': True,
                'message': f'Speech saved to {output_file}',
                'file_path': output_file
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def speech_to_text(audio_file):
    """
    Converts speech to text (requires speech_recognition library)
    
    Args:
        audio_file (str): Path to audio file
        
    Returns:
        dict: Transcribed text or error
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return {
            'success': False,
            'error': 'speech_recognition library not installed. Run: pip install SpeechRecognition'
        }
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        return {
            'success': True,
            'text': text
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
