from data import TEXT_COLUMN
import pandas as pd
from constants import DATA_FT
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')

fdist = FreqDist()

try:
    print(f"Loading data from '{DATA_FT}' ... ", end='')
    df = pd.read_feather(DATA_FT)

    for text in df[TEXT_COLUMN].values:
        for word in word_tokenize(text):
            fdist[word] += 1

    for k, v in fdist.items():
        print(f"{k}: {v}")

except:
    print('No data found')
