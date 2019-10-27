"""Utility functions
"""

import re

import nltk
from constants import COMMON_WORDS
from nltk.corpus import stopwords

nltk.download('stopwords')

EXCLUDED_WORDS = []
EXCLUDED_WORDS.extend(stopwords.words('english'))
EXCLUDED_WORDS.extend(stopwords.words('swedish'))
EXCLUDED_WORDS.extend(COMMON_WORDS)

BAD_SYMBOLS_RE = re.compile('[^0-9a-zåäöéè]')

def clean_text(text):
    """ text: a string
        return: modified initial string
    """
    text = str(text).lower()  # lowercase text
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in EXCLUDED_WORDS)
    return text
