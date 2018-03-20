# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Unit tests for artifact conversion to and from Tensorflow SavedModel."""

import tensorflow as tf
import unittest
import glob
import json
import os
import shutil
import tempfile

import convert

SAVED_MODEL_DIR = 'saved_model'


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    def create_saved_model(self):
        x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
        w = tf.Variable(tf.random_uniform([2, 2]))
        y = tf.matmul(x, w)
        output = tf.nn.softmax(y)
        init_op = w.initializer

        # Create a builder
        builder = tf.saved_model.builder.SavedModelBuilder(
            os.path.join(self._tmp_dir, SAVED_MODEL_DIR))

        with tf.Session() as sess:
            # Run the initializer on `w`.
            sess.run(init_op)

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map=None,
                assets_collection=None)

        builder.save()

    def test_convert_saved_model(self):
        self.create_saved_model()

        convert.convert('Softmax',
                        os.path.join(self._tmp_dir, SAVED_MODEL_DIR,
                                     'converted_model.pb'), 'serve',
                        os.path.join(self._tmp_dir, SAVED_MODEL_DIR))

        weights = [{
            'paths': ['group1-shard1of1'],
            'weights': [{
                'shape': [2, 2],
                'name': 'Softmax',
                'dtype': 'float32'
            }]
        }]
        # Load the saved weights as a JSON string.
        weights_manifest = open(
            os.path.join(self._tmp_dir, SAVED_MODEL_DIR,
                         'weights_manifest.json'), 'rt')
        output_json = json.load(weights_manifest)
        weights_manifest.close()
        self.assertEqual(output_json, weights)

        # Check the content of the output directory.
        self.assertTrue(
            glob.glob(
                os.path.join(self._tmp_dir, SAVED_MODEL_DIR,
                             'converted_model.pb')))
        self.assertTrue(
            glob.glob(
                os.path.join(self._tmp_dir, SAVED_MODEL_DIR, 'group*-*')))


if __name__ == '__main__':
    unittest.main()
