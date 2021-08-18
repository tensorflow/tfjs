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
"""Unit tests for artifact conversion to and from Python tf.keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflowjs import version
from tensorflowjs.converters import converter
from tensorflowjs.converters import keras_tfjs_loader


# TODO(adarob): Add tests for quantization option.


class ConvertH5WeightsTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(ConvertH5WeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertH5WeightsTest, self).tearDown()

  def testWeightsOnly(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      input_tensor = tf.keras.layers.Input((3,))
      dense1 = tf.keras.layers.Dense(
          4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
          name='MyDense1')(input_tensor)
      output = tf.keras.layers.Dense(
          2, use_bias=False, kernel_initializer='ones', name='MyDense2')(dense1)
      model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output])
      h5_path = os.path.join(self._tmp_dir, 'MyModel.h5')
      model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    model_json, groups = (
        converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
            h5_path, output_dir=self._tmp_dir))
    self.assertIsNone(model_json)

    # Check the loaded weights.
    self.assertEqual(1, len(groups))
    self.assertEqual(3, len(groups[0]))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertSavedKerasModelNoSplitByLayer(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      input_tensor = tf.keras.layers.Input((3,))
      dense1 = tf.keras.layers.Dense(
          4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
          name='MergedDense1')(input_tensor)
      output = tf.keras.layers.Dense(
          2, use_bias=False,
          kernel_initializer='ones', name='MergedDense2')(dense1)
      model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output])
      h5_path = os.path.join(self._tmp_dir, 'MyModelMerged.h5')
      model.save(h5_path)

    # Load the saved weights as a JSON string.
    model_json, groups = (
        converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
            h5_path, output_dir=self._tmp_dir))
    # check the model topology was stored
    self.assertIsInstance(model_json['model_config'], dict)
    self.assertIsInstance(model_json['model_config']['config'], dict)
    self.assertIn('layers', model_json['model_config']['config'])

    # Check the loaded weights.
    self.assertEqual(tf.keras.__version__, model_json['keras_version'])
    self.assertEqual('tensorflow', model_json['backend'])
    self.assertEqual(1, len(groups))
    self.assertEqual(3, len(groups[0]))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertSavedKerasModelSplitByLayer(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      input_tensor = tf.keras.layers.Input((3,))
      dense1 = tf.keras.layers.Dense(
          4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
          name='MergedDense1')(input_tensor)
      output = tf.keras.layers.Dense(
          2, use_bias=False,
          kernel_initializer='ones', name='MergedDense2')(dense1)
      model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output])
      h5_path = os.path.join(self._tmp_dir, 'MyModelMerged.h5')
      model.save(h5_path)

    # Load the saved weights as a JSON string.
    model_json, groups = (
        converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
            h5_path, output_dir=self._tmp_dir, split_weights_by_layer=True))
    # check the model topology was stored
    self.assertIsInstance(model_json['model_config'], dict)
    self.assertIsInstance(model_json['model_config']['config'], dict)
    self.assertIn('layers', model_json['model_config']['config'])

    # Check the loaded weights.
    self.assertEqual(tf.keras.__version__, model_json['keras_version'])
    self.assertEqual('tensorflow', model_json['backend'])
    self.assertEqual(2, len(groups))
    self.assertEqual(2, len(groups[0]))
    self.assertEqual(1, len(groups[1]))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testConvertSavedKerasModeltoTfLayersModelSharded(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save(h5_path)

      weights = sequential_model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      # Due to the shard size, there ought to be 4 shards after conversion.
      weight_shard_size_bytes = int(total_weight_bytes * 0.3)

      # Convert Keras model to tfjs_layers_model format.
      output_dir = os.path.join(self._tmp_dir, 'sharded_tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, output_dir,
          weight_shard_size_bytes=weight_shard_size_bytes)

      weight_files = sorted(glob.glob(os.path.join(output_dir, 'group*.bin')))
      self.assertEqual(len(weight_files), 4)
      weight_file_sizes = [os.path.getsize(f) for f in weight_files]
      self.assertEqual(sum(weight_file_sizes), total_weight_bytes)
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[1])
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[2])
      self.assertLess(weight_file_sizes[3], weight_file_sizes[0])

  def testConvertWeightsFromSequentialModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    model_json, groups = (
        converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
            h5_path, output_dir=self._tmp_dir))
    self.assertIsNone(model_json)

    # Check the loaded weights.
    self.assertEqual(1, len(groups))
    self.assertEqual(3, len(groups[0]))
    # contents of weights are verified in tests of the library code

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(model_json, output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testUserDefinedMetadata(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)

    metadata_json = {'a': 1}
    converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
        h5_path, output_dir=self._tmp_dir, metadata={'key': metadata_json})

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))
    self.assertEqual(metadata_json, output_json['userDefinedMetadata']['key'])

  def testConvertModelForNonexistentDirCreatesDir(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      output_dir = os.path.join(self._tmp_dir, 'foo_model')
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, output_dir=output_dir)

    # Check the content of the output directory.
    output_json = json.load(
        open(os.path.join(output_dir, 'model.json'), 'rt'))
    self.assertIsNone(output_json['modelTopology'])
    self.assertIsInstance(output_json['weightsManifest'], list)
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testOutpuDirAsAnExistingFileLeadsToValueError(self):
    output_path = os.path.join(self._tmp_dir, 'foo_model')
    with open(output_path, 'wt') as f:
      f.write('\n')

    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save_weights(h5_path)

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'already exists as a file'):
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, output_dir=output_path)

  def testTensorflowjsToKerasConversionSucceeds(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save(h5_path)
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, output_dir=self._tmp_dir)
      old_model_json = sequential_model.to_json()

    # Convert the tensorflowjs artifacts to a new H5 file.
    new_h5_path = os.path.join(self._tmp_dir, 'new.h5')
    converter.dispatch_tensorflowjs_to_keras_h5_conversion(
        os.path.join(self._tmp_dir, 'model.json'), new_h5_path)

    # Load the new H5 and compare the model JSONs.
    with tf.Graph().as_default(), tf.compat.v1.Session():
      new_model = tf.keras.models.load_model(new_h5_path)
      self.assertEqual(old_model_json, new_model.to_json())

  def testTensorflowjsToKerasConversionFailsOnDirInputPath(self):
    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'input path should be a model\.json file'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          self._tmp_dir, os.path.join(self._tmp_dir, 'new.h5'))

  def testTensorflowjsToKerasConversionFailsOnExistingDirOutputPath(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      sequential_model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(
              3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
              name='Dense1'),
          tf.keras.layers.Dense(
              1, use_bias=False, kernel_initializer='ones', name='Dense2')])
      h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
      sequential_model.save(h5_path)
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, output_dir=self._tmp_dir)

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'but received an existing directory'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          os.path.join(self._tmp_dir, 'model.json'), self._tmp_dir)

  def testTensorflowjsToKerasConversionFailsOnInvalidJsonFile(self):
    fake_json_path = os.path.join(self._tmp_dir, 'fake.json')
    with open(fake_json_path, 'wt') as f:
      f.write('__invalid_json_content__')

    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'cannot read valid JSON content from'):
      converter.dispatch_tensorflowjs_to_keras_h5_conversion(
          fake_json_path, os.path.join(self._tmp_dir, 'model.h5'))


class ConvertKerasToTfGraphModelTest(tf.test.TestCase):

  def setUp(self):
    super(ConvertKerasToTfGraphModelTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertKerasToTfGraphModelTest, self).tearDown()

  def testConvertKerasModelToTfGraphModel(self):
    output_dir = os.path.join(self._tmp_dir, 'foo_model')
    sequential_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense1')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save(h5_path)
    converter.dispatch_keras_h5_to_tfjs_graph_model_conversion(
        h5_path, output_dir=output_dir)

    # Check model.json and weights manifest.
    with open(os.path.join(output_dir, 'model.json'), 'rt') as f:
      model_json = json.load(f)
    self.assertTrue(model_json['modelTopology'])
    self.assertIsNot(model_json['modelTopology']['versions'], None)
    signature = model_json['signature']
    self.assertIsNot(signature, None)
    self.assertIsNot(signature['inputs'], None)
    self.assertIsNot(signature['outputs'], None)
    weights_manifest = model_json['weightsManifest']
    self.assertEqual(len(weights_manifest), 1)
    # Check meta-data in the artifact JSON.
    self.assertEqual(model_json['format'], 'graph-model')
    self.assertEqual(
        model_json['convertedBy'],
        'TensorFlow.js Converter v%s' % version.version)
    self.assertEqual(model_json['generatedBy'],
                     tf.__version__)
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testConvertKerasModelToTfGraphModelSharded(self):
    output_dir = os.path.join(self._tmp_dir, 'foo_model')
    sequential_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense1')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save(h5_path)

    # Do initial conversion without sharding.
    converter.dispatch_keras_h5_to_tfjs_graph_model_conversion(
        h5_path, output_dir)
    weight_files = glob.glob(os.path.join(output_dir, 'group*.bin'))

    # Get size of weights in bytes after graph optimizations.
    optimized_total_weight = sum([os.path.getsize(f) for f in weight_files])

    # Due to the shard size, there ought to be 4 shards after conversion.
    weight_shard_size_bytes = int(optimized_total_weight * 0.3)

    output_dir = os.path.join(self._tmp_dir, 'sharded_model')
    # Convert Keras model again with shard argument set.
    converter.dispatch_keras_h5_to_tfjs_graph_model_conversion(
        h5_path, output_dir,
        weight_shard_size_bytes=weight_shard_size_bytes)

    weight_files = sorted(glob.glob(os.path.join(output_dir, 'group*.bin')))
    self.assertEqual(len(weight_files), 4)
    weight_file_sizes = [os.path.getsize(f) for f in weight_files]
    self.assertEqual(sum(weight_file_sizes), optimized_total_weight)
    self.assertEqual(weight_file_sizes[0], weight_file_sizes[1])
    self.assertEqual(weight_file_sizes[0], weight_file_sizes[2])
    self.assertLess(weight_file_sizes[3], weight_file_sizes[0])

  def testUserDefinedMetadata(self):
    output_dir = os.path.join(self._tmp_dir, 'foo_model')
    sequential_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense1')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save(h5_path)
    metadata_json = {'a': 1}
    converter.dispatch_keras_h5_to_tfjs_graph_model_conversion(
        h5_path, output_dir=output_dir, metadata={'key': metadata_json})

    # Check model.json and weights manifest.
    with open(os.path.join(output_dir, 'model.json'), 'rt') as f:
      model_json = json.load(f)
    self.assertEqual(metadata_json, model_json['userDefinedMetadata']['key'])

class ConvertTfKerasSavedModelTest(tf.test.TestCase):

  def setUp(self):
    super(ConvertTfKerasSavedModelTest, self).setUp()
    self._tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertTfKerasSavedModelTest, self).tearDown()

  def _createSimpleSequentialModel(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape([2, 3], input_shape=[6]))
    model.add(tf.keras.layers.LSTM(10))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.predict(tf.ones((1, 6)), steps=1)
    tf.keras.backend.set_learning_phase(0)
    return model

  def _createNestedSequentialModel(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(6, input_shape=[10], activation='relu'))
    model.add(self._createSimpleSequentialModel())
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.predict(tf.ones((1, 10)), steps=1)
    return model

  def _createFunctionalModelWithWeights(self):
    input1 = tf.keras.Input(shape=[8])
    input2 = tf.keras.Input(shape=[10])
    y = tf.keras.layers.Concatenate()([input1, input2])
    y = tf.keras.layers.Dense(4, activation='softmax')(y)
    model = tf.keras.Model([input1, input2], y)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.predict([tf.ones((1, 8)), tf.ones((1, 10))], steps=1)
    return model

  def testConvertTfKerasSequentialSavedAsSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      old_model_json = json.loads(model.to_json())
      old_weights = model.get_weights()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
          self._tmp_dir, tfjs_output_dir)

      # Verify the size of the weight file.
      weight_path = glob.glob(os.path.join(tfjs_output_dir, 'group*-*'))[0]
      weight_file_bytes = os.path.getsize(weight_path)
      model_weight_bytes = sum(w.size * 4 for w in model.get_weights())
      self.assertEqual(weight_file_bytes, model_weight_bytes)

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # Load the converted mode back.
      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      model_prime = keras_tfjs_loader.load_keras_model(model_json_path)
      new_weights = model_prime.get_weights()

      # Check the equality of the old and new model JSONs.
      self.assertEqual(old_model_json, json.loads(model_prime.to_json()))

      # Check the equality of the old and new weights.
      self.assertAllClose(old_weights, new_weights)

  def testConvertTfKerasSequentialCompiledAndSavedAsSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()

      old_model_json = json.loads(model.to_json())
      old_weights = model.get_weights()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
          self._tmp_dir, tfjs_output_dir)

      # Verify the size of the weight file.
      weight_path = glob.glob(os.path.join(tfjs_output_dir, 'group*-*'))[0]
      weight_file_bytes = os.path.getsize(weight_path)
      model_weight_bytes = sum(w.size * 4 for w in model.get_weights())
      self.assertEqual(weight_file_bytes, model_weight_bytes)

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # Load the converted mode back.
      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      model_prime = keras_tfjs_loader.load_keras_model(model_json_path)
      new_weights = model_prime.get_weights()

      # Check the equality of the old and new model JSONs.
      self.assertEqual(old_model_json, json.loads(model_prime.to_json()))

      # Check the equality of the old and new weights.
      self.assertAllClose(old_weights, new_weights)

  def testWrongConverterRaisesCorrectErrorMessage(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Use wrong dispatcher.
      with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
          ValueError,
          r'Expected path to point to an HDF5 file, but it points to a '
          r'directory'):
        converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
            self._tmp_dir, tfjs_output_dir)

  def testConvertTfKerasNestedSequentialSavedAsSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createNestedSequentialModel()
      old_model_json = json.loads(model.to_json())
      old_weights = model.get_weights()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
          self._tmp_dir, tfjs_output_dir)

      # Verify the size of the weight file.
      weight_path = glob.glob(os.path.join(tfjs_output_dir, 'group*-*'))[0]
      weight_file_bytes = os.path.getsize(weight_path)
      model_weight_bytes = sum(w.size * 4 for w in model.get_weights())
      self.assertEqual(weight_file_bytes, model_weight_bytes)

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # Load the converted mode back.
      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      model_prime = keras_tfjs_loader.load_keras_model(model_json_path)
      new_weights = model_prime.get_weights()

      # Check the equality of the old and new model JSONs.
      self.assertEqual(old_model_json, json.loads(model_prime.to_json()))

      # Check the equality of the old and new weights.
      self.assertAllClose(old_weights, new_weights)

  def testConvertTfKerasFunctionalModelWithWeightsSavedAsSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createFunctionalModelWithWeights()
      old_model_json = json.loads(model.to_json())
      old_weights = model.get_weights()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
          self._tmp_dir, tfjs_output_dir)

      # Verify the size of the weight file.
      weight_path = glob.glob(os.path.join(tfjs_output_dir, 'group*-*'))[0]
      weight_file_bytes = os.path.getsize(weight_path)
      model_weight_bytes = sum(w.size * 4 for w in model.get_weights())
      self.assertEqual(weight_file_bytes, model_weight_bytes)

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # Load the converted mode back.
      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      model_prime = keras_tfjs_loader.load_keras_model(model_json_path)
      new_weights = model_prime.get_weights()

      # Check the equality of the old and new model JSONs.
      self.assertEqual(old_model_json, json.loads(model_prime.to_json()))

      # Check the equality of the old and new weights.
      self.assertAllClose(old_weights, new_weights)

  def testConvertTfKerasSequentialSavedAsSavedModelWithQuantization(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      tf.keras.models.save_model(model, self._tmp_dir, save_format='tf')

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_saved_model_to_tensorflowjs_conversion(
          self._tmp_dir, tfjs_output_dir,
          quantization_dtype_map={'uint16': '*'})

      # Verify the size of the weight file.
      weight_path = glob.glob(os.path.join(tfjs_output_dir, 'group*-*'))[0]
      weight_file_bytes = os.path.getsize(weight_path)
      # Each uint16 number has 2 bytes.
      bytes_per_num = 2
      model_weight_bytes = sum(
          w.size * bytes_per_num for w in model.get_weights())
      self.assertEqual(weight_file_bytes, model_weight_bytes)

  def testConvertTfjsLayersModelToShardedWeights(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

      weight_shard_size_bytes = int(total_weight_bytes * 0.3)
      # Due to the shard size, there ought to be 4 shards after conversion.

      # Convert the tfjs model to another tfjs model, with a specified weight
      # shard size.
      sharded_model_path = os.path.join(self._tmp_dir, 'sharded_model')
      converter.dispatch_tensorflowjs_to_tensorflowjs_conversion(
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_path,
          weight_shard_size_bytes=weight_shard_size_bytes)

      # Check the number of sharded files and their sizes.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_path, 'group*.bin')))
      self.assertEqual(len(weight_files), 4)
      weight_file_sizes = [os.path.getsize(f) for f in weight_files]
      self.assertEqual(sum(weight_file_sizes), total_weight_bytes)
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[1])
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[2])
      self.assertLess(weight_file_sizes[3], weight_file_sizes[0])

  def testConvertTfjsLayersModelWithShardSizeGreaterThanTotalWeightSize(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

      weight_shard_size_bytes = int(total_weight_bytes * 2)
      # Due to the shard size, there ought to be 1 shard after conversion.

      # Convert the tfjs model to another tfjs model, with a specified weight
      # shard size.
      sharded_model_path = os.path.join(self._tmp_dir, 'sharded_model')
      converter.dispatch_tensorflowjs_to_tensorflowjs_conversion(
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_path,
          weight_shard_size_bytes=weight_shard_size_bytes)

      # Check the number of sharded files and their sizes.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_path, 'group*.bin')))
      self.assertEqual(len(weight_files), 1)
      weight_file_sizes = [os.path.getsize(f) for f in weight_files]
      self.assertEqual(sum(weight_file_sizes), total_weight_bytes)

  def testTfjsLayer2TfjsLayersConversionWithExistingFilePathFails(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

      # Convert the tfjs model to another tfjs model, with a specified weight
      # shard size.
      sharded_model_path = os.path.join(self._tmp_dir, 'sharded_model')
      with open(sharded_model_path, 'wt') as f:
        # Create a fie at the path to elicit the error.
        f.write('hello')
      with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
          ValueError, r'already exists as a file'):
        converter.dispatch_tensorflowjs_to_tensorflowjs_conversion(
            os.path.join(tfjs_output_dir, 'model.json'), sharded_model_path)

  def testConvertTfjsLayersModelWithUint16Quantization(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

      weight_shard_size_bytes = int(total_weight_bytes * 2)
      # Due to the shard size, there ought to be 1 shard after conversion.

      # Convert the tfjs model to another tfjs model, with quantization.
      sharded_model_path = os.path.join(self._tmp_dir, 'sharded_model')
      converter.dispatch_tensorflowjs_to_tensorflowjs_conversion(
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_path,
          quantization_dtype_map={'uint16': '*'},
          weight_shard_size_bytes=weight_shard_size_bytes)

      # Check the number of quantized files and their sizes.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_path, 'group*.bin')))
      self.assertEqual(len(weight_files), 1)
      weight_file_size = os.path.getsize(weight_files[0])

      # The size of the saved weight file should reflect the result of the
      # uint16 quantization.
      self.assertEqual(weight_file_size, total_weight_bytes / 2)

  def testConvertTfjsLayersModelWithUint8Quantization(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()
      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

      weight_shard_size_bytes = int(total_weight_bytes * 2)
      # Due to the shard size, there ought to be 1 shard after conversion.

      # Convert the tfjs model to another tfjs model, with quantization.
      sharded_model_path = os.path.join(self._tmp_dir, 'sharded_model')
      converter.dispatch_tensorflowjs_to_tensorflowjs_conversion(
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_path,
          quantization_dtype_map={'uint8': '*'},
          weight_shard_size_bytes=weight_shard_size_bytes)

      # Check the number of quantized files and their sizes.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_path, 'group*.bin')))
      self.assertEqual(len(weight_files), 1)
      weight_file_size = os.path.getsize(weight_files[0])

      # The size of the saved weight file should reflect the result of the
      # uint16 quantization.
      self.assertEqual(weight_file_size, total_weight_bytes / 4)

  def testConvertTfjsLayersModelToKerasSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = self._createSimpleSequentialModel()

      # Save the keras model to a .h5 file.
      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path)

      # Convert the keras SavedModel to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      converter.dispatch_keras_h5_to_tfjs_layers_model_conversion(
          h5_path, tfjs_output_dir)

    # Convert the tfjs LayersModel to tf.keras SavedModel.
    keras_saved_model_dir = os.path.join(self._tmp_dir, 'saved_model')
    converter.dispatch_tensorflowjs_to_keras_saved_model_conversion(
        os.path.join(tfjs_output_dir, 'model.json'), keras_saved_model_dir)

    # Check the files of the keras SavedModel.
    files = glob.glob(os.path.join(keras_saved_model_dir, '*'))
    self.assertIn(os.path.join(keras_saved_model_dir, 'saved_model.pb'), files)
    self.assertIn(os.path.join(keras_saved_model_dir, 'variables'), files)
    self.assertIn(os.path.join(keras_saved_model_dir, 'assets'), files)


if __name__ == '__main__':
  tf.test.main()
