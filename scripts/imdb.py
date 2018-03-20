# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""IMDB sentiment classification example.

Based on Python Keras examples:
  https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
  https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py

TODO(cais): Add
  https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
  once b/74429960 is fixed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import keras
import tensorflowjs as tfjs


INDEX_FROM = 3
# Offset in word index. Used during word index lookup and reverse lookup.


def get_word_index(reverse=False):
  """Get word index.

  Args:
    reverse: Reverse the index, so that the returned index is from index values
      to words.

  Returns:
    The word index as a `dict`.
  """
  word_index = keras.datasets.imdb.get_word_index()
  if reverse:
    word_index = dict((word_index[key], key) for key in word_index)
  return word_index


def indices_to_words(reverse_index, indices):
  """Convert an iterable of word indices into words.

  Args:
    reverse_index: An `dict` mapping word index (as `int`) to word (as `str`).
    indices: An iterable of word indices.

  Returns:
    Mapped words as a `list` of `str`s.
  """
  return [reverse_index[i - INDEX_FROM] if i >= INDEX_FROM else 'OOV'
          for i in indices]


def get_imdb_data(vocabulary_size, max_len):
  """Get IMDB data for training and validation.

  Args:
    vocabulary_size: Size of the vocabulary, as an `int`.
    max_len: Cut text after this number of words.

  Returns:
    x_train: An int array of shape `(num_exapmles, max_len)`: index-encoded
      sentences.
    y_train: An int array of shape `(num_exapmles,)`: labels for the sentences.
    x_test: Same as `x_train`, but for test.
    y_test: Same as `y_train`, but for test.
  """
  print("Getting IMDB data with vocabulary_size %d" % vocabulary_size)
  (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
      num_words=vocabulary_size)
  x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
  x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
  return x_train, y_train, x_test, y_test


def train_model(model_type,
                vocabulary_size,
                embedding_size,
                x_train,
                y_train,
                x_test,
                y_test,
                epochs,
                batch_size):
  """Train a model for IMDB sentiment classification.

  Args:
    model_type: Type of the model to train, as a `str`.
    vocabulary_size: Vocabulary size.
    embedding_size: Embedding dimensions.
    x_train: An int array of shape `(num_exapmles, max_len)`: index-encoded
      sentences.
    y_train: An int array of shape `(num_exapmles,)`: labels for the sentences.
    x_test: Same as `x_train`, but for test.
    y_test: Same as `y_train`, but for test.
    epochs: Number of epochs to train the model for.
    batch_size: Batch size to use during trainng.

  Returns:
    The trained model instance.

  Raises:
    ValueError: on invalid model type.
  """

  model = keras.Sequential()
  model.add(keras.layers.Embedding(vocabulary_size, embedding_size))
  if model_type == 'bidirectional_lstm':
    # TODO(cais): Uncomment the following once bug b/74429960 is fixed.
    # model.add(keras.layers.Embedding(
    #     vocabulary_size, 128, input_length=maxlen))
    # model.add(keras.layers.Bidirectional(
    #     keras.layers.LSTM(64,
    #                       kernel_initializer='glorot_normal',
    #                       recurrent_initializer ='glorot_normal')))
    # model.add(keras.layers.Dropout(0.5))
    raise NotImplementedError()
  elif model_type == 'cnn':
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(250,
                                  3,
                                  padding='valid',
                                  activation='relu',
                                  strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(250, activation='relu'))
  elif model_type == 'lstm':
    model.add(
        keras.layers.LSTM(
            128,
            kernel_initializer='glorot_normal',
            recurrent_initializer='glorot_normal'))
    # TODO(cais): Remove glorot_normal and use the default orthogonal once
    #   SVD is available.
  else:
    raise ValueError("Invalid model type: '%s'" % model_type)
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[x_test, y_test])
  return model


def main():
  x_train, y_train, x_test, y_test = (
      get_imdb_data(FLAGS.vocabulary_size, FLAGS.max_len))

  model = train_model(FLAGS.model_type,
                      FLAGS.vocabulary_size,
                      FLAGS.embedding_size,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      FLAGS.epochs,
                      FLAGS.batch_size)

  # Display a number test phrases and their final classification.
  forward_index = get_word_index()
  reverse_index = get_word_index(reverse=True)
  print('\n')
  for i in range(FLAGS.num_show):
    print('--- Test Case %d ---' % (i + 1))
    print('Sentence: "' +
          ' '.join(indices_to_words(reverse_index, x_test[i, :])) + '"')
    print('Truth: %d' % y_test[i])
    print('Prediction: %s\n' % model.predict(x_test[i : i + 1, :])[0][0])

  # Save metadata, including word index, INDEX_FROM and max_len and model
  # hyperparameters.
  metadata = {
      'word_index': forward_index,
      'index_from': INDEX_FROM,
      'max_len': FLAGS.max_len,
      'model_type': FLAGS.model_type,
      'vocabulary_size': FLAGS.vocabulary_size,
      'embedding_size': FLAGS.embedding_size,
      'epochs': FLAGS.epochs,
      'batch_size': FLAGS.batch_size,
  }

  if not os.path.isdir(FLAGS.artifacts_dir):
    os.makedirs(FLAGS.artifacts_dir)
  metadata_json_path = os.path.join(FLAGS.artifacts_dir, 'imdb.metadata.json')
  json.dump(metadata, open(metadata_json_path, 'wt'))
  print('\nSaved model metadata at: %s' % metadata_json_path)

  tfjs.converters.save_keras_model(model, FLAGS.artifacts_dir)
  print('\nSaved model artifcats in directory: %s' % FLAGS.artifacts_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('IMDB sentiment classification model')
  parser.add_argument(
      'model_type',
      type=str,
      help='Type of model to train for the IMDB sentiment classification task: '
      '(cnn | lstm)')
  parser.add_argument(
      '--vocabulary_size',
      type=int,
      default=20000,
      help='Vocabulary size.')
  parser.add_argument(
      '--embedding_size',
      type=int,
      default=128,
      help='Embedding size.')
  parser.add_argument(
      '--max_len',
      type=int,
      default=100,
      help='Cut text after this number of words.')
  parser.add_argument(
      '--epochs',
      type=int,
      default=5,
      help='Number of epochs to train the model for.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size used during training.')
  parser.add_argument(
      '--num_show',
      type=int,
      default=5,
      help='Number of sentences to show prediction score on after training.')
  parser.add_argument(
      '--artifacts_dir',
      type=str,
      default='/tmp/mnist.keras',
      help='Local path for saving the TensorFlow.js artifacts.')

  FLAGS, _ = parser.parse_known_args()
  main()
