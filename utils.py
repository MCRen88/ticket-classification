"""Utility functions
"""

import re

from nltk.corpus import stopwords

from constants import COMMON_WORDS

EXCLUDED_WORDS = []
EXCLUDED_WORDS.extend(stopwords.words('english'))
EXCLUDED_WORDS.extend(stopwords.words('swedish'))
EXCLUDED_WORDS.extend(COMMON_WORDS)
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')


def clean_text(text):
    """ text: a string
        return: modified initial string
    """
    text = str(text).lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    words = text.split(' ')
    words = [w.strip() for w in words if w.isalpha()]
    words = [w for w in words if w not in EXCLUDED_WORDS]

    text = ' '.join(words)
    return text
