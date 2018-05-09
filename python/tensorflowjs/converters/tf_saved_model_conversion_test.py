# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for artifact conversion to and from Tensorflow SavedModel."""

import glob
import json
import os
import shutil
import tempfile
import unittest

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import tensorflow_hub as hub
from tensorflowjs.converters import tf_saved_model_conversion

SAVED_MODEL_DIR = 'saved_model'
SESSION_BUNDLE_MODEL_DIR = 'session_bundle'
FROZEN_MODEL_DIR = 'frozen_model'
HUB_MODULE_DIR = 'hub_module'


class ConvertTest(unittest.TestCase):
  def setUp(self):
    super(ConvertTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertTest, self).tearDown()

  def create_session_bundle(self):
    graph = tf.Graph()
    with graph.as_default():
      x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
      w = tf.Variable(tf.random_uniform([2, 2]))
      y = tf.matmul(x, w)
      softmax = tf.nn.softmax(y)
      init_op = w.initializer

      # Create a builder
      saver = tf.train.Saver()

      with tf.Session() as sess:
        # Run the initializer on `w`.
        sess.run(init_op)
        softmax.op.run()
        saver.save(sess, os.path.join(
            self._tmp_dir, SESSION_BUNDLE_MODEL_DIR, 'model'))

  def create_saved_model(self):
    graph = tf.Graph()
    with graph.as_default():
      x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
      w = tf.Variable(tf.random_uniform([2, 2]))
      y = tf.matmul(x, w)
      tf.nn.softmax(y)
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

  def create_hub_module(self):
    # Module function that doubles its input.
    def double_module_fn():
      w = tf.Variable([2.0, 4.0])
      x = tf.placeholder(dtype=tf.float32)
      hub.add_signature(inputs=x, outputs=x*w)
    graph = tf.Graph()
    with graph.as_default():
      spec = hub.create_module_spec(double_module_fn)
      m = hub.Module(spec)
    # Export the module.
    with tf.Session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      m.export(os.path.join(self._tmp_dir, HUB_MODULE_DIR), sess)

  def create_frozen_model(self):
    graph = tf.Graph()
    saved_model_dir = os.path.join(self._tmp_dir, FROZEN_MODEL_DIR)
    with graph.as_default():
      x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
      w = tf.Variable(tf.random_uniform([2, 2]))
      y = tf.matmul(x, w)
      tf.nn.softmax(y)
      init_op = w.initializer

      # Create a builder
      builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

      with tf.Session() as sess:
        # Run the initializer on `w`.
        sess.run(init_op)

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=None,
            assets_collection=None)

      builder.save()

    frozen_file = os.path.join(self._tmp_dir, FROZEN_MODEL_DIR, 'model.frozen')
    freeze_graph.freeze_graph(
        '',
        '',
        True,
        '',
        "Softmax",
        '',
        '',
        frozen_file,
        True,
        '',
        saved_model_tags=tf.saved_model.tag_constants.SERVING,
        input_saved_model_dir=saved_model_dir)

  def test_convert_saved_model(self):
    self.create_saved_model()
    print(glob.glob(
        os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR, '*')))

    tf_saved_model_conversion.convert_tf_saved_model(
        os.path.join(self._tmp_dir, SAVED_MODEL_DIR),
        'Softmax',
        os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    )

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
                         'tensorflowjs_model.pb')))
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, SAVED_MODEL_DIR, 'group*-*')))

  def test_convert_session_bundle(self):
    self.create_session_bundle()

    tf_saved_model_conversion.convert_tf_session_bundle(
        os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR),
        'Softmax',
        os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR)
    )

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
        os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR,
                     'weights_manifest.json'), 'rt')
    output_json = json.load(weights_manifest)
    weights_manifest.close()
    self.assertEqual(output_json, weights)
    # Check the content of the output directory.
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR,
                         'tensorflowjs_model.pb')))
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, SESSION_BUNDLE_MODEL_DIR, 'group*-*')))

  def test_convert_frozen_model(self):
    self.create_frozen_model()
    print(glob.glob(
        os.path.join(self._tmp_dir, FROZEN_MODEL_DIR, '*')))

    tf_saved_model_conversion.convert_tf_frozen_model(
        os.path.join(self._tmp_dir, FROZEN_MODEL_DIR, 'model.frozen'),
        'Softmax',
        os.path.join(self._tmp_dir, FROZEN_MODEL_DIR))

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
        os.path.join(self._tmp_dir, FROZEN_MODEL_DIR,
                     'weights_manifest.json'), 'rt')
    output_json = json.load(weights_manifest)
    weights_manifest.close()
    self.assertEqual(output_json, weights)

    # Check the content of the output directory.
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, FROZEN_MODEL_DIR,
                         'tensorflowjs_model.pb')))
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, FROZEN_MODEL_DIR, 'group*-*')))

  def test_convert_hub_module(self):
    self.create_hub_module()
    print(glob.glob(
        os.path.join(self._tmp_dir, HUB_MODULE_DIR, '*')))

    tf_saved_model_conversion.convert_tf_hub_module(
        os.path.join(self._tmp_dir, HUB_MODULE_DIR),
        os.path.join(self._tmp_dir, SAVED_MODEL_DIR),
        'default'
    )

    weights = [{
        'paths': ['group1-shard1of1'],
        'weights': [{
            'shape': [2],
            'name': 'module/Variable',
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
                         'tensorflowjs_model.pb')))
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, SAVED_MODEL_DIR, 'group*-*')))


if __name__ == '__main__':
  unittest.main()
