# AUTO-DOWNLOAD REQUIRED NLTK DATA
import nltk


# auto download code here

import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# your full code...

# Download only if missing (safe for all machines)
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")






# df = pd.read_csv("data/news.csv")
# documents = df["content"].astype(str).tolist()
# src/preprocess.py
# src/preprocess.py
# Basic preprocessing utilities for the project.
# Student-style, simple and clear.
# AI-assisted: I used AI guidance to sketch a robust but simple edit-distance
# spell-correction helper. Include disclosure in report.

# REST OF YOUR PREPROCESSING CODE
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Run these once in your environment if not done already:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# small manual normalization dictionary (student-provided)
NORMALIZE_MAP = {
    "covid19": "covid",
    "covid-19": "covid",
    "u.s.": "usa",
    "u.s": "usa",
    "pak": "pakistan",
    "ai": "artificial intelligence"
}

def basic_clean(text):
    """Lowercase, remove non-alphanumeric chars (keep spaces), collapse spaces."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)       # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # keep letters/numbers/spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Split on whitespace (simple tokenizer)."""
    text = basic_clean(text)
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def lemmatize(tokens):
    return [LEMMATIZER.lemmatize(t) for t in tokens]

def normalize_tokens(tokens):
    """Apply manual normalizations (map tokens using NORMALIZE_MAP)."""
    out = []
    for t in tokens:
        out.append(NORMALIZE_MAP.get(t, t))
    return out

# --- Spell correction (small helper) ---
# Note: full edit-distance based spell-checking is long to implement robustly.
# I used AI to help design this simple suggestion function. Keep disclosure.
def edits1(word):
    """Return all strings that are one edit away from `word` (basic)."""
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def known(words, vocabulary):
    return set(w for w in words if w in vocabulary)

def simple_spell_correct(token, vocabulary):
    """
    Very simple spelling correction:
    - if token in vocabulary -> return token
    - else check edits1 candidates present in vocabulary -> pick any
    - else return original token
    (This is intentionally simple: manual corrections + small dictionary are better.)
    """
    if token in vocabulary:
        return token
    cand = known(edits1(token), vocabulary)
    if cand:
        # pick candidate with smallest edit distance/shortest as tie-breaker
        return sorted(list(cand), key=lambda x: (len(x), x))[0]
    return token

def expand_with_wordnet(token):
    """Return small set of synonyms for token using WordNet (for expansion)."""
    syns = set()
    for syn in wordnet.synsets(token):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name != token:
                syns.add(name)
    return list(syns)

def preprocess_text(text, vocabulary=None, do_spell_correct=False):
    """Full student pipeline for a single document/query."""
    toks = tokenize(text)
    if do_spell_correct and vocabulary is not None:
        toks = [simple_spell_correct(t, vocabulary) for t in toks]
    toks = remove_stopwords(toks)
    toks = lemmatize(toks)
    toks = normalize_tokens(toks)
    return toks
