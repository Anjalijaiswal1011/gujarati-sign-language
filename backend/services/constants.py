"""
Central constants – labels, thresholds, and Gujarati vocabulary.

Gujarati Sign Language (GSL) word set — 30 most common words
used in daily communication, education, and assistive tech.
"""

# ── ASL / GSL Alphabet Labels (A-Z + 3 special) ─────────────────────────────
ALPHABET_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

# ── Gujarati Sign Language — 30 Core Vocabulary Words ───────────────────────
# These are the words the LSTM model will classify
WORDS_LABELS = [
    "hello",        # નમસ્તે
    "thank_you",    # આભાર
    "yes",          # હા
    "no",           # ના
    "please",       # કૃપા
    "sorry",        # માફ
    "help",         # મદદ
    "water",        # પાણી
    "food",         # ખોરાક
    "home",         # ઘર
    "school",       # શાળા
    "doctor",       # ડૉક્ટર
    "mother",       # માતા
    "father",       # પિતા
    "family",       # પરિવાર
    "friend",       # મિત્ર
    "good",         # સારું
    "bad",          # ખરાબ
    "love",         # પ્રેમ
    "eat",          # ખાઓ
    "drink",        # પીઓ
    "come",         # આવો
    "go",           # જાઓ
    "stop",         # રોકો
    "name",         # નામ
    "my",           # મારું
    "your",         # તમારું
    "how_are_you",  # કેમ છો
    "i_am_fine",    # હું ઠીક છું
    "what",         # શું
]

# ── Gujarati Translation Map (English word → Gujarati script) ────────────────
GUJARATI_WORD_MAP = {
    # Alphabet letters (phonetic Gujarati)
    "A": "અ", "B": "બ", "C": "ક", "D": "ડ", "E": "ઇ",
    "F": "ફ", "G": "ગ", "H": "હ", "I": "ઈ", "J": "જ",
    "K": "ક", "L": "લ", "M": "મ", "N": "ન", "O": "ઓ",
    "P": "પ", "Q": "ક", "R": "ર", "S": "સ", "T": "ટ",
    "U": "ઉ", "V": "વ", "W": "ડ", "X": "ક", "Y": "ય", "Z": "ઝ",

    # Words (Gujarati direct translations)
    "hello": "હેલો",
    "thank_you": "આભાર",
    "yes": "હા",
    "no": "ના",
    "please": "કૃપા કરીને",
    "sorry": "માફ કરશો",
    "help": "મદદ",
    "water": "પાણી",
    "food": "ખોરાક",
    "home": "ઘર",
    "school": "શાળા",
    "doctor": "ડૉક્ટર",
    "mother": "માતા",
    "father": "પિતા",
    "family": "પરિવાર",
    "friend": "મિત્ર",
    "good": "સારું",
    "bad": "ખરાબ",
    "love": "પ્રેમ",
    "eat": "ખાઓ",
    "drink": "પીઓ",
    "come": "આવો",
    "go": "જાઓ",
    "stop": "રોકો",
    "name": "નામ",
    "my": "મારું",
    "your": "તમારું",
    "how_are_you": "કેમ છો?",
    "i_am_fine": "હું ઠીક છું",
    "what": "શું",

    # Additional common phrases
    "thank": "આભાર",
    "namaste": "નમસ્તે",
    "kem_cho": "કેમ છો?",
    "maja_ma": "મજામાં",
    "hu_thik_chhu": "હું ઠીક છું",
    "shu_che": "શું છે?",
    "maru_nam": "મારું નામ",
    "tamaru_nam": "તમારું નામ",
    "pan": "પણ",
    "che": "છે",
    "hu": "હું",
    "ame": "અમે",
    "tame": "તમે",
    "aa": "આ",
    "te": "તે",
    "pan_che": "પણ છે",
    "kai_nathi": "કઈ નથી",
    "thayo": "થયો",
    "aav": "આવ",
    "ja": "જા",
    "beso": "બેસો",
    "uth": "ઉઠ",
    "kha": "ખા",
    "pi": "પી",
    "school": "શાળા",
    "hospital": "દવાખાનું",
    "police": "પોલીસ",
    "fire": "આગ",
    "ambulance": "એમ્બ્યુલન્સ",
    "pain": "દર્દ",
    "sick": "બીમાર",
    "medicine": "દવા",
    "money": "પૈસા",
    "work": "કામ",
    "time": "સમય",
    "day": "દિવસ",
    "night": "રાત",
    "morning": "સવાર",
    "evening": "સાંજ",
    "today": "આજ",
    "tomorrow": "કાલ",
    "yesterday": "ગઈ કાલ",
    "book": "ચોપડી",
    "pen": "કલમ",
    "table": "ટેબલ",
    "chair": "ખુરશી",
    "door": "દરવાજો",
    "window": "બારી",
    "phone": "ફોન",
    "computer": "કોમ્પ્યુટર",
    "car": "ગાડી",
    "bus": "બસ",
    "train": "ટ્રેન",
    "walk": "ચાલો",
    "run": "દોડો",
    "sit": "બેસો",
    "stand": "ઊભા",
    "sleep": "સૂઓ",
    "wake": "ઉઠો",
    "big": "મોટું",
    "small": "નાનું",
    "hot": "ગરમ",
    "cold": "ઠંડું",
    "fast": "ઝડપ",
    "slow": "ધીમું",
    "new": "નવું",
    "old": "જૂનું",
    "happy": "ખુશ",
    "sad": "દુઃખી",
    "angry": "ગુસ્સો",
    "tired": "થાક",
    "hungry": "ભૂખ",
    "thirsty": "તરસ",
    "beautiful": "સુંદર",
    "clean": "સ્વચ્છ",
    "dirty": "ગંદું",
    "open": "ખોલો",
    "close": "બંધ",
    "give": "આપો",
    "take": "લો",
    "buy": "ખરીદો",
    "sell": "વેચો",
    "read": "વાંચો",
    "write": "લખો",
    "speak": "બોલો",
    "listen": "સાંભળો",
    "see": "જુઓ",
    "know": "જાણો",
    "understand": "સમજો",
    "remember": "યાદ",
    "forget": "ભૂલ",
    "learn": "શીખો",
    "teach": "શીખવો",
    "play": "રમો",
    "sing": "ગાઓ",
    "dance": "નૃત્ય",
    "cook": "રાંધો",
    "wash": "ધઓ",
    "wear": "પહેરો",
    "remove": "કાઢો",
    "pray": "પ્રાર્થના",
    "respect": "આદર",
    "congratulations": "અભિનંદન",
    "welcome": "સ્વાગત",
    "goodbye": "આવજો",
    "ok": "ઠીક",
    "fine": "ઠીક",
    "not_ok": "ઠીક નહીં",
    "one": "એક", "two": "બે", "three": "ત્રણ",
    "four": "ચાર", "five": "પાંચ", "six": "છ",
    "seven": "સાત", "eight": "આઠ", "nine": "નવ", "ten": "દસ",
    # Phonetic Multilingual Mapping for Hindi/English sign gestures
    "dhanyawad": "આભાર",
    "shukriya": "આભાર",
    "paani": "પાણી",
    "jal": "પાણી",
    "haa": "હા",
    "naa": "ના",
    "aao": "આવો",
    "ruko": "રોકો",
    "khana": "ખાઓ",
    "peena": "પીઓ",
    "madad": "મદદ",
}

# ── Confidence thresholds ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 60.0   # Minimum % to accept alphabet prediction
WORD_CONFIDENCE_THRESHOLD = 45.0  # Minimum % for word/LSTM prediction
STABLE_FRAMES_REQUIRED  = 5      # Same gesture must appear this many times
WORD_STABLE_FRAMES      = 3      # Frames for word gesture stabilization

# ── Model paths ──────────────────────────────────────────────────────────────
ALPHABET_MODEL_PATH  = "services/models/model1/keras_model.h5"
ALPHABET_LABELS_PATH = "services/models/model1/labels.txt"
WORDS_MODEL_PATH     = "services/models/Model2/keras_model.h5"
WORDS_LABELS_PATH    = "services/models/Model2/labels.txt"

# ── Recognition modes ─────────────────────────────────────────────────────────
MODE_ALPHABET = "alphabet"
MODE_WORDS    = "words"
DEFAULT_MODE  = MODE_ALPHABET
