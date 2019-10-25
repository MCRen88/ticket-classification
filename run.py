
import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp

from constants import DATA_PATH, EPOCHS, SKIP_PROJECTS
from data import Data
from models import LSTMClassifier

data = Data(path=DATA_PATH, skip_projects=SKIP_PROJECTS)

data.preview()
data.save_distribution('distribution.pdf')
data.print_labels()

# Hyperparameters
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([5]))
HP_EMBEDDING_SIZE = hp.HParam('emb_size', hp.Discrete([100, 200]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128]))
HP_LSTM_SIZE = hp.HParam('lstm_size', hp.Discrete([100]))
HP_MAX_NB_WORDS = hp.HParam(
    'max_nb_words', hp.Discrete([50000]))
HP_MAX_SEQ_LEN = hp.HParam(
    'max_seq_len', hp.Discrete([20, 50, 100, 200, 500]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_EPOCHS, HP_EMBEDDING_SIZE, HP_LSTM_SIZE,
                 HP_BATCH_SIZE, HP_MAX_NB_WORDS, HP_MAX_SEQ_LEN],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def run(log_dir, hparams):
    """Run training
    """
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)

        X, Y = data.datasets(max_sequence_length=hparams[HP_MAX_SEQ_LEN],
                             max_nb_words=hparams[HP_MAX_NB_WORDS])

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.10, random_state=42)
        nb_features, nb_classes = X.shape[1], Y.shape[1]

        model = LSTMClassifier(
            nb_features, nb_classes, x_test, y_test, log_dir,
            emb_size=hparams[HP_EMBEDDING_SIZE],
            lstm_size=hparams[HP_LSTM_SIZE],
            max_sequence_length=hparams[HP_MAX_SEQ_LEN],
            max_nb_words=hparams[HP_MAX_NB_WORDS])

        print(model.summary())

        model.train(x_train, y_train,
                    epochs=EPOCHS, batch_size=hparams[HP_BATCH_SIZE])

        accuracy = model.evaluate(x_test, y_test)

        tf.summary.scalar(METRIC_ACCURACY, accuracy[1], step=1)

        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
            accuracy[0], accuracy[1]))


for emb_size in HP_EMBEDDING_SIZE.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for lstm_size in HP_LSTM_SIZE.domain.values:
            for max_seq_len in HP_MAX_SEQ_LEN.domain.values:
                for max_nb_words in HP_MAX_NB_WORDS.domain.values:

                    hparams = {
                        HP_EMBEDDING_SIZE: emb_size,
                        HP_BATCH_SIZE: batch_size,
                        HP_LSTM_SIZE: lstm_size,
                        HP_MAX_SEQ_LEN: max_seq_len,
                        HP_MAX_NB_WORDS: max_nb_words,
                        HP_EPOCHS: EPOCHS
                    }

                    run_name = f"run-{datetime.datetime.now():%Y%m%d-%H%M%S}"
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams)
