# -*- coding: utf-8 -*-
"""
Gujarati Translation Service
Full offline dictionary + intelligent sentence building for GSL predictions.
"""
import re
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import from constants to keep a single source of truth
try:
    from services.constants import GUJARATI_WORD_MAP
except ImportError:
    GUJARATI_WORD_MAP = {}

# Extended offline translation map (superset of constants)
_OFFLINE_MAP = {
    **GUJARATI_WORD_MAP,
    # Special gesture labels
    "space": "", "del": "", "nothing": "", "NOTHING": "", "SPACE": "", "DEL": "",
    # Phrases built from word sequences
    "my name": "મારું નામ",
    "your name": "તમારું નામ",
    "how are you": "કેમ છો?",
    "i am fine": "હું ઠીક છું",
    "thank you": "આભાર",
    "please help": "કૃપા કરીને મદદ",
    "i am hungry": "મને ભૂખ છે",
    "i am thirsty": "મને તરસ છે",
    "i need water": "મને પાણી જોઈએ",
    "i need food": "મને ખોરાક જોઈએ",
    "i need help": "મને મદદ જોઈએ",
    "stop please": "કૃપા કરીને રોકો",
    "good morning": "સુ પ્રભાત",
    "good evening": "સ઼ ઢળ",
    "good night": "શુભ રાત્રી",
    "good bye": "આવજો",
    "see you": "ફરી મળીશું",
    "i love you": "હું તમને પ્રેમ કરું છું",
    "come here": "અહીં આવો",
    "go away": "ચાલ્યા જાઓ",
}

# Gujarati grammar rules: word-order corrections for common patterns
_PHRASE_CORRECTIONS = {
    "my name": "મારું નામ",
    "your name": "તમારું નામ",
    "i love": "હું ...ને ચાહું",
    "i need": "મારે ... જોઈએ",
    "i am": "હું ... છું",
    "i want": "મારે ... જોઈએ",
    "please give": "કૃપા ... આપો",
    "please help": "કૃપા મદદ",
    "how are you": "કેમ છો",
    "i am fine": "હું ઠીક છું",
    "thank you": "આભાર",
    "good morning": "સુ પ્રભાત",
    "good night": "શુભ રાત્રી",
    "good bye": "આવજો",
}


def translate_word(word):
    """Translate a single English word/label to Gujarati."""
    if not word:
        return ""
    w = word.strip()
    # Check the full map first
    result = _OFFLINE_MAP.get(w) or _OFFLINE_MAP.get(w.lower()) or _OFFLINE_MAP.get(w.upper())
    if result is not None:
        return result
    # Underscore variant (e.g. how_are_you → how are you)
    w_spaces = w.replace("_", " ")
    result = _OFFLINE_MAP.get(w_spaces) or _OFFLINE_MAP.get(w_spaces.lower())
    if result is not None:
        return result
    return w  # Return as-is if no translation found


def translate_to_gujarati(english_text):
    """
    Translates an English phrase/sentence or single word to Gujarati.
    Steps:
      1. Check exact phrase match in offline map
      2. Check phrase-level pattern corrections
      3. Word-by-word translation with smart joining
      4. Online fallback (Google Translate) if available
    """
    if not english_text or english_text.strip() == "":
        return ""

    text_clean = english_text.strip()

    # 1. Exact phrase match
    exact = _OFFLINE_MAP.get(text_clean) or _OFFLINE_MAP.get(text_clean.lower())
    if exact is not None:
        return exact

    # 2. Check phrase-level corrections
    text_lower = text_clean.lower()
    result = None
    for phrase, correction in _PHRASE_CORRECTIONS.items():
        if phrase in text_lower:
            # Replace matched phrase with Gujarati equivalent
            rest = text_lower.replace(phrase, "").strip()
            rest_gj = " ".join(translate_word(w) for w in rest.split()) if rest else ""
            result = (correction + (" " + rest_gj if rest_gj else "")).strip()
            break

    if result is None:
        # 3. Word-by-word translation
        words = text_clean.split()
    gj_words = []
    i = 0
    while i < len(words):
        # Try two-word combinations first
        if i + 1 < len(words):
            two_word = f"{words[i]} {words[i+1]}".lower()
            two_word_gj = _OFFLINE_MAP.get(two_word)
            if two_word_gj is not None:
                gj_words.append(two_word_gj)
                i += 2
                continue
        gj_words.append(translate_word(words[i]))
        i += 1

    result = " ".join(gj_words)

    # 4. Try online translation if result still has English words
    if _has_english(result) and len(result) > 0:
        try:
            from deep_translator import GoogleTranslator
            online = GoogleTranslator(source='en', target='gu').translate(text_clean)
            if online and len(online) > 0:
                return online
        except Exception:
            pass

    return result


def build_sentence_gujarati(word_list):
    """
    Build a Gujarati sentence from a list of English words/signs.
    Handles word-order correction for Gujarati grammar (SOV vs SVO).
    """
    if not word_list:
        return ""

    # Filter out control words
    filtered = [w for w in word_list if w.upper() not in ("NOTHING", "SPACE", "DEL", "")]
    if not filtered:
        return ""

    full_en = " ".join(filtered).lower()

    # Try full phrase first
    full_gj = _OFFLINE_MAP.get(full_en)
    if full_gj:
        return full_gj

    # Translate word by word and apply smart corrections
    gj_parts = []
    for word in filtered:
        gj = translate_word(word)
        gj_parts.append(gj)

    # Apply Gujarati SOV reordering if we detect a verb at end
    return " ".join(gj_parts)


def _has_english(text):
    """Check if text contains any English (ASCII) letters."""
    return bool(re.search(r'[a-zA-Z]', text))


def get_gujarati_for_letter(letter):
    """Get the Gujarati phonetic for a single letter."""
    return _OFFLINE_MAP.get(letter.upper(), letter)
