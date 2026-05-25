from services.translation_service import translate_to_gujarati

GLOSS_MAP = {
    ("hello",): "Hello!",
    ("thank", "you"): "Thank you!",
    ("sorry",): "I am sorry.",
    ("yes",): "Yes.",
    ("no",): "No.",
    ("hello", "name", "what"): "Hello, what is your name?",
    ("how", "are", "you"): "How are you?",
    ("water", "please"): "Can I have some water, please?",
    ("food", "please"): "Can I have some food, please?",
    ("i", "need", "help"): "I need help.",
    ("where", "hospital"): "Where is the hospital?",
    ("go", "home"): "I want to go home.",
    ("i", "love", "family"): "I love my family.",
    ("you", "deaf"): "Are you deaf?",
    ("good", "morning"): "Good morning.",
    ("doctor", "where"): "Where is the doctor?",
    ("i", "want", "sleep"): "I want to sleep.",
    ("school", "go"): "Let's go to school."
}

class NLPProcessor:
    @staticmethod
    def correct_sentence(words_list):
        """
        Takes a list of words, cleans them up, matches them to natural English sentences,
        and translates the corrected sentence to Gujarati.
        
        Returns:
            dict: {
                "original_words": "word1 word2...",
                "corrected_english": "Cleaned sentence...",
                "gujarati_translation": "ગુજરાતી ભાષાંતર..."
            }
        """
        if not words_list:
            return {
                "original_words": "",
                "corrected_english": "",
                "gujarati_translation": ""
            }
            
        # Clean consecutive duplicates
        cleaned_words = []
        for w in words_list:
            w_clean = w.strip().lower()
            if not w_clean:
                continue
            if not cleaned_words or cleaned_words[-1] != w_clean:
                cleaned_words.append(w_clean)
                
        if not cleaned_words:
            return {
                "original_words": "",
                "corrected_english": "",
                "gujarati_translation": ""
            }
            
        # Try matching predefined gloss patterns
        words_tuple = tuple(cleaned_words)
        corrected_en = None
        
        if words_tuple in GLOSS_MAP:
            corrected_en = GLOSS_MAP[words_tuple]
        else:
            # Simple heuristic reconstruction
            sentence_parts = []
            for w in cleaned_words:
                if w == "i":
                    sentence_parts.append("I")
                else:
                    sentence_parts.append(w)
            
            raw_sentence = " ".join(sentence_parts)
            # Basic sentence casing and punctuation
            corrected_en = raw_sentence[0].upper() + raw_sentence[1:]
            
            # Simple question heuristic
            question_words = ["what", "where", "how", "who", "why", "when", "are", "is"]
            if cleaned_words[0] in question_words or (len(cleaned_words) > 1 and cleaned_words[-1] in question_words):
                corrected_en += "?"
            else:
                corrected_en += "."
                
        # Translate to Gujarati
        gujarati_tr = translate_to_gujarati(corrected_en)
        
        return {
            "original_words": " ".join(words_list),
            "corrected_english": corrected_en,
            "gujarati_translation": gujarati_tr
        }
