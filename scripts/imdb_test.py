# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Test for the IMDB model and supporting functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import unittest

import numpy as np
import keras

from scripts import imdb


class IMDBTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(IMDBTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(IMDBTest, self).tearDown()

  def testGetWordIndexForward(self):
    word_index = imdb.get_word_index()
    self.assertGreater(word_index['bar'], 0)

  def testIndicesToWordsReverse(self):
    forward_index = imdb.get_word_index()
    reverse_index = imdb.get_word_index(reverse=True)
    self.assertEqual('bar', reverse_index[forward_index['bar']])

  def testIndicesToWords(self):
    forward_index = imdb.get_word_index()
    reverse_index = imdb.get_word_index(reverse=True)
    indices = [forward_index[word] + imdb.INDEX_FROM
               for word in ['one', 'two', 'three']]
    self.assertEqual(['one', 'two', 'three'],
                     imdb.indices_to_words(reverse_index, indices))

  def testTrainLSTMModel(self):
    data_size = 10
    x_train = np.random.randint(0, 100, (data_size,))
    y_train = np.random.randint(0, 2, (data_size,))
    x_test = np.random.randint(0, 100, (data_size,))
    y_test = np.random.randint(0, 2, (data_size,))

    vocabulary_size = 100
    embedding_size = 32
    epochs = 1
    batch_size = data_size
    model = imdb.train_model(
        'lstm', vocabulary_size, embedding_size,
        x_train, y_train, x_test, y_test,
        epochs, batch_size)
    self.assertTrue(model.layers)

  def testTrainModelWithInvalidModelTypeRaisesError(self):
    data_size = 10
    x_train = np.random.randint(0, 100, (data_size,))
    y_train = np.random.randint(0, 2, (data_size,))
    x_test = np.random.randint(0, 100, (data_size,))
    y_test = np.random.randint(0, 2, (data_size,))

    vocabulary_size = 100
    embedding_size = 32
    epochs = 1
    batch_size = data_size
    with self.assertRaises(ValueError):
      imdb.train_model(
          'nonsensical_model_type', vocabulary_size, embedding_size,
          x_train, y_train, x_test, y_test,
          epochs, batch_size)


if __name__ == '__main__':
  unittest.main()
