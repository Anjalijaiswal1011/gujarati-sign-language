"""
INSTALLATION AND SETUP GUIDE
Hand Detection Dataset, Voice Conversion & Gujarati Language Support
"""

# =============================================================================
# 1. INSTALL REQUIRED PACKAGES
# =============================================================================

INSTALLATION_COMMANDS = {
    'voice_conversion': [
        'pip install gtts',                    # Google Text-to-Speech
        'pip install SpeechRecognition',       # Speech Recognition
        'pip install pydub',                   # Audio processing
    ],
    'hand_detection': [
        'pip install mediapipe',               # Pre-trained hand detection (recommended)
        'pip install opencv-python',           # Already installed
        'pip install ultralytics',             # YOLO hand detection (alternative)
    ],
    'gujarati_translation': [
        'pip install deep-translator',         # Already installed (supports Gujarati)
        'pip install textblob',                # Language detection
        'pip install langdetect',              # Alternative language detection
    ],
}

# =============================================================================
# 2. QUICK START - INSTALL ALL
# =============================================================================

"""
Run this command in PowerShell:

pip install gtts SpeechRecognition pydub mediapipe ultralytics textblob langdetect

OR individually:

# For Voice Conversion (Text-to-Speech & Speech-to-Text)
pip install gtts SpeechRecognition pydub

# For Hand Detection (MediaPipe - recommended for quick start)
pip install mediapipe

# For Gujarati Translation (already supported via deep-translator)
pip install textblob langdetect
"""

# =============================================================================
# 3. API ENDPOINTS - NEW FEATURES
# =============================================================================

API_ENDPOINTS = {
    'text_to_speech_gujarati': {
        'endpoint': 'POST /api/translator/voice/text-to-speech/',
        'description': 'Convert text to speech in Gujarati or other languages',
        'request': {
            'text': 'String to convert',
            'language': 'gu (Gujarati), en (English), etc.'
        },
        'example': """
        curl -X POST http://localhost:8000/api/translator/voice/text-to-speech/ \\
            -H "Content-Type: application/json" \\
            -d '{"text": "નમસ્તે", "language": "gu"}'
        """
    },
    'speech_to_text': {
        'endpoint': 'POST /api/translator/voice/speech-to-text/',
        'description': 'Convert speech to text',
        'request': 'Upload audio file',
        'example': """
        curl -X POST http://localhost:8000/api/translator/voice/speech-to-text/ \\
            -F "audio=@/path/to/audio.wav"
        """
    },
    'enhanced_translate': {
        'endpoint': 'POST /api/translator/enhanced/translate/',
        'description': 'Translate text with full language support including Gujarati',
        'request': {
            'text': 'Hello world',
            'source_language': 'en',
            'target_language': 'gu'
        },
        'example': """
        curl -X POST http://localhost:8000/api/translator/enhanced/translate/ \\
            -H "Content-Type: application/json" \\
            -d '{"text": "Hello", "source_language": "en", "target_language": "gu"}'
        """
    },
    'batch_translate': {
        'endpoint': 'POST /api/translator/batch/',
        'description': 'Translate multiple texts at once',
        'request': {
            'texts': ['Text1', 'Text2'],
            'source_language': 'en',
            'target_language': 'gu'
        }
    },
    'supported_languages': {
        'endpoint': 'GET /api/translator/languages/',
        'description': 'Get list of supported languages',
        'example': 'curl http://localhost:8000/api/translator/languages/'
    },
    'detect_language': {
        'endpoint': 'POST /api/translator/detect/',
        'description': 'Detect language of input text',
        'request': {'text': 'Text to detect'},
        'example': """
        curl -X POST http://localhost:8000/api/translator/detect/ \\
            -H "Content-Type: application/json" \\
            -d '{"text": "Namaste"}'
        """
    }
}

# =============================================================================
# 4. HAND DETECTION DATASETS
# =============================================================================

HAND_DETECTION_OPTIONS = {
    'mediapipe_prebuilt': {
        'description': 'Google MediaPipe pre-trained hand detection',
        'pros': ['Easy to use', 'Pre-trained', 'High accuracy', 'Real-time'],
        'cons': ['Limited customization'],
        'installation': 'pip install mediapipe',
        'code_example': """
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        results = hands.process(image)
        """
    },
    'yolov8_custom': {
        'description': 'YOLOv8 with custom hand dataset',
        'pros': ['Customizable', 'High accuracy', 'Fast'],
        'cons': ['Requires training dataset'],
        'installation': 'pip install ultralytics',
        'code_example': """
        from ultralytics import YOLO
        
        model = YOLO('yolov8n.pt')
        results = model.predict(source='image.jpg')
        """
    },
    'custom_training': {
        'description': 'Train your own model',
        'pros': ['Full control', 'Domain-specific accuracy'],
        'cons': ['Requires large dataset', 'Training time'],
        'datasets': [
            'Egohands: http://vision.soic.indiana.edu/projects/egohands/',
            'Cornell Hand: http://pr.cs.cornell.edu/grasping/',
            'Kaggle: https://www.kaggle.com/search?q=hand+gesture'
        ]
    }
}

# =============================================================================
# 5. GUJARATI LANGUAGE SUPPORT
# =============================================================================

GUJARATI_SUPPORT = {
    'text_translation': {
        'service': 'deep-translator with Google Translate',
        'language_code': 'gu',
        'example': """
        from services.enhanced_translation_service import translate_text
        
        result = translate_text(
            text='Hello world',
            source_lang='en',
            target_lang='gu'  # Gujarati
        )
        print(result['translated_text'])  # આવો પ્રાર્થના કરો
        """
    },
    'text_to_speech_gujarati': {
        'service': 'Google Text-to-Speech (gTTS)',
        'language_code': 'gu',
        'example': """
        from services.voice_converter import text_to_speech
        
        result = text_to_speech(
            text='નમસ્તે',  # Namaste in Gujarati
            language='gu',
            output_file='namaste.mp3'
        )
        """
    },
    'speech_to_text': {
        'service': 'Google Speech Recognition',
        'example': """
        from services.voice_converter import speech_to_text
        
        result = speech_to_text('audio_gujarati.wav')
        print(result['text'])  # Recognized Gujarati text
        """
    }
}

# =============================================================================
# 6. USAGE EXAMPLES
# =============================================================================

USAGE_EXAMPLES = {
    'python_backend': """
# Text-to-Speech Gujarati
from services.voice_converter import text_to_speech
text_to_speech('નમસ્તે', language='gu', output_file='namaste.mp3')

# Enhanced Translation
from services.enhanced_translation_service import translate_text
result = translate_text('Hello', source_lang='en', target_lang='gu')

# Hand Detection
from services.hand_detection_datasets import check_mediapipe_installation
status = check_mediapipe_installation()
print(status)

# Batch Translation
from services.enhanced_translation_service import translate_batch
results = translate_batch(
    ['Hi', 'Good morning', 'How are you?'],
    source_lang='en',
    target_lang='gu'
)
    """,
    
    'javascript_frontend': """
// Text-to-Speech API
async function textToSpeech(text, language = 'gu') {
    const response = await fetch('/api/translator/voice/text-to-speech/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, language})
    });
    return await response.json();
}

// Enhanced Translation API
async function translateText(text, target = 'gu') {
    const response = await fetch('/api/translator/enhanced/translate/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text,
            source_language: 'en',
            target_language: target
        })
    });
    return await response.json();
}

// Speech-to-Text API
async function speechToText(audioFile) {
    const formData = new FormData();
    formData.append('audio', audioFile);
    const response = await fetch('/api/translator/voice/speech-to-text/', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}

// Supported Languages
async function getLanguages() {
    const response = await fetch('/api/translator/languages/');
    return await response.json();
}

// Usage
textToSpeech('નમસ્તે').then(result => console.log(result));
translateText('Hello world', 'gu').then(result => console.log(result));
    """
}

# =============================================================================
# 7. TROUBLESHOOTING
# =============================================================================

TROUBLESHOOTING = {
    'gtts_not_found': {
        'error': 'ModuleNotFoundError: No module named \'gtts\'',
        'solution': 'pip install gtts'
    },
    'mediapipe_not_found': {
        'error': 'ModuleNotFoundError: No module named \'mediapipe\'',
        'solution': 'pip install mediapipe'
    },
    'speech_recognition_failed': {
        'error': 'Speech recognition returns empty',
        'solution': [
            'Check microphone is working',
            'Ensure audio quality is good',
            'Install: pip install pydub'
        ]
    },
    'translation_network_error': {
        'error': 'ConnectionError during translation',
        'solution': 'Check internet connection, try again'
    }
}

# =============================================================================
# 8. FULL INSTALLATION COMMAND
# =============================================================================

print("""
╔════════════════════════════════════════════════════════════════════╗
║  COMPLETE INSTALLATION FOR ALL FEATURES                           ║
╚════════════════════════════════════════════════════════════════════╝

Run in PowerShell:

pip install gtts SpeechRecognition pydub mediapipe ultralytics textblob langdetect

Then restart your Django server:
cd c:\\Users\\chandrashekhar\\OneDrive\\Desktop\\minor\\backend
python manage.py runserver 0.0.0.0:8000

Test endpoints:
curl http://localhost:8000/api/translator/languages/

✓ Features Added:
  • Hand Detection (MediaPipe + YOLO option)
  • Text-to-Speech in Gujarati & other languages
  • Speech-to-Text conversion
  • Enhanced translation with batch support
  • Language detection
  • Gujarati language full support

✓ New API Endpoints:
  POST /api/translator/voice/text-to-speech/
  POST /api/translator/voice/speech-to-text/
  POST /api/translator/enhanced/translate/
  POST /api/translator/batch/
  GET /api/translator/languages/
  POST /api/translator/detect/
""")
