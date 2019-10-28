
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from constants import DATA_FT
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from utils import clean_text

TEXT_COLUMN = 'text'
PROJECT_COLUMN = 'project'


class Data:
    """Data holder
    """

    def __init__(self, path=None, skip_projects=None):
        self.path = path
        self.skip_projects = skip_projects
        self.label_encoder = LabelEncoder()

        self._load()
        self.labels = self.label_encoder.fit_transform(self.df[PROJECT_COLUMN])

    def _load(self):
        """Load data from file-system or from previously stored data.
        """
        try:
            print(f"Loading data from '{DATA_FT}' ... ", end='')
            self.df = pd.read_feather(DATA_FT)

        except:
            file_paths = []
            for root, _, f_names in os.walk(self.path):
                if len(f_names) == 0:
                    continue
                for f in f_names:
                    file_paths.append(os.path.join(root, f))

            d = {PROJECT_COLUMN: [], TEXT_COLUMN: []}

            count = 0
            for file_path in file_paths:
                with open(file_path) as file:

                    info = json.loads(file.read())

                    project = info['project']['INFO_HEAD']
                    if not project:
                        continue
                    if project in self.skip_projects:
                        continue

                    title = info['title']
                    description = info['description']
                    text = clean_text(f"{title} {description}")

                    d[TEXT_COLUMN].append(text)
                    d[PROJECT_COLUMN].append(project)

                    count += 1
                    print(
                        f"\rParsing {count}/{len(file_paths)} files ... ", end='')

            self.df = pd.DataFrame(data=d)
            self.df.to_feather(DATA_FT)

        print('DONE')

    def preview(self):
        """Preview the first 10 lines of dataset
        """
        print(self.df.head(10))

    def save_distribution(self, filename=None):
        """Save the data distribution to a PDF file
        """
        self.df[PROJECT_COLUMN].value_counts().plot(kind='barh').invert_yaxis()
        plt.yticks(fontsize=7)
        plt.xlabel("Number of issues")
        plt.savefig(filename, bbox_inches='tight')

    def datasets(self, max_sequence_length=None, max_nb_words=None):
        """Create datasets
        """
        tokenizer = Tokenizer(num_words=max_nb_words)
        tokenizer.fit_on_texts(self.df[TEXT_COLUMN].values)

        X = tokenizer.texts_to_sequences(self.df[TEXT_COLUMN].values)
        X = pad_sequences(X, maxlen=max_sequence_length)

        Y = pd.get_dummies(self.labels).values

        return X, Y

    def print_labels(self):
        """Print the name of labels
        """
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{i}\t{label}")
