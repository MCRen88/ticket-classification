
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import LambdaCallback, TensorBoard
from keras.layers import (LSTM, Bidirectional, Dense, Embedding,
                          SpatialDropout1D)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_sample_weight


class Classifier(Sequential):
    """Base class for classifiers
    """

    def __init__(self, nb_features, nb_classes,
                 x_test, y_test, log_dir, **kwargs):
        super().__init__()
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.x_test = x_test
        self.y_test = y_test
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(self.log_dir + '/cm')

    def train(self, x_train, y_train, epochs, batch_size):
        """Train model
        """
        tensorboard_callback = TensorBoard(
            log_dir=self.log_dir, update_freq='batch', profile_batch=0)
        cm_callback = LambdaCallback(
            on_epoch_end=self.log_confusion_matrix)
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        return self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        class_weight=weights, validation_split=0.1,
                        callbacks=[tensorboard_callback, cm_callback],
                        validation_data=(self.x_test, self.y_test))

    def test(self, text, max_sequence_length=None, max_nb_words=None):
        """Predict category text
        """
        tokenizer = Tokenizer(num_words=max_nb_words)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_sequence_length)
        return self.predict(padded)

    def log_confusion_matrix(self, epoch, logs):
        """ Use the model to predict the values from the validation dataset.
        """
        classes = range(self.nb_classes)
        test_pred = self.predict_classes(self.x_test)

        con_mat = tf.math.confusion_matrix(
            labels=self.y_test.argmax(axis=1),
            predictions=test_pred).numpy()
        con_mat_norm = np.around(con_mat.astype(
            'float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_df = pd.DataFrame(con_mat_norm,
                                  index=classes,
                                  columns=classes)

        figure, ax = plt.subplots(figsize=(20, 16))
        sns.set(font_scale=1.4)
        sns.heatmap(con_mat_df, cmap="binary")
        plt.ylabel('True project')
        plt.xlabel('Predicted project')
        ax.set_ylim(self.nb_classes, 0)  # clipping bug

        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)


class LSTMClassifier(Classifier):
    """LSTM classifier
    """

    def __init__(self, nb_features, nb_classes,
                 x_test, y_test, log_dir, **kwargs):
        super().__init__(nb_features, nb_classes,
                         x_test, y_test, log_dir, **kwargs)

        lstm_size = kwargs.get('lstm_size')
        max_nb_words = kwargs.get('max_nb_words')
        emb_size = kwargs.get('emb_size')

        self.add(Embedding(max_nb_words, emb_size,
                           input_length=self.nb_features))
        self.add(SpatialDropout1D(0.5))
        self.add(LSTM(lstm_size, dropout=0.5, recurrent_dropout=0.5))
        self.add(Dense(self.nb_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


class LSTMBidirectionalClassifier(Classifier):
    """Bidirecitonal LSTM
    """

    def __init__(self, nb_features, nb_classes,
                 x_test, y_test, log_dir, **kwargs):
        super().__init__(nb_features, nb_classes,
                         x_test, y_test, log_dir, **kwargs)

        lstm_size = kwargs.get('lstm_size')
        max_nb_words = kwargs.get('max_nb_words')
        emb_size = kwargs.get('emb_size')

        self.add(
            Embedding(max_nb_words, emb_size,
                      input_length=self.nb_features))
        self.add(SpatialDropout1D(0.2))
        self.add(Bidirectional(
            LSTM(lstm_size, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat'))
        self.add(Dense(self.nb_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
