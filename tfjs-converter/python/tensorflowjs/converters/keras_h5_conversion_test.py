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
"""Unit tests for artifact conversion to and from Python keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflowjs import version
from tensorflowjs.converters import keras_h5_conversion as conversion


class ConvertH5WeightsTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(ConvertH5WeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ConvertH5WeightsTest, self).tearDown()

  def testConvertWeightsFromSimpleModelNoSplitByLayer(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MyDense10')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False, kernel_initializer='ones', name='MyDense20')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModel.h5')
    model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    groups = conversion.h5_weights_to_tfjs_format(h5py.File(h5_path))

    # Check the loaded weights.
    # Due to the default `split_by_layer=True`, there should be only one weight
    # group.
    self.assertEqual(1, len(groups))
    self.assertEqual(3, len(groups[0]))
    kernel1 = groups[0][0]
    self.assertEqual('MyDense10/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((3, 4), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 4]), kernel1['data']))
    bias1 = groups[0][1]
    self.assertEqual('MyDense10/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((4,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([4]), bias1['data']))
    kernel2 = groups[0][2]
    self.assertEqual('MyDense20/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((4, 2), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([4, 2]), kernel2['data']))

  def testConvertWeightsFromSimpleModelSplitByLayer(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MyDense30')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False, kernel_initializer='ones', name='MyDense40')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModel.h5')
    model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    groups = conversion.h5_weights_to_tfjs_format(h5py.File(h5_path),
                                                  split_by_layer=True)

    # Check the loaded weights.
    # Due to `split_by_layer=True` and the fact that the model has two layers,
    # there should be two weight groups.
    self.assertEqual(2, len(groups))
    self.assertEqual(2, len(groups[0]))
    kernel1 = groups[0][0]
    self.assertEqual('MyDense30/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((3, 4), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 4]), kernel1['data']))
    bias1 = groups[0][1]
    self.assertEqual('MyDense30/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((4,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([4]), bias1['data']))

    self.assertEqual(1, len(groups[1]))
    kernel2 = groups[1][0]
    self.assertEqual('MyDense40/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((4, 2), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([4, 2]), kernel2['data']))

  def testConvertModelWithNestedLayerNames(self):
    model = keras.Sequential()

    # Add a layer with a nested layer name, i.e., a layer name with slash(es)
    # in it.
    model.add(keras.layers.Dense(2, input_shape=[12], name='dense'))
    model.add(keras.layers.Dense(8, name='foo/dense'))
    model.add(keras.layers.Dense(4, name='foo/bar/dense'))
    tfjs_path = os.path.join(self._tmp_dir, 'nested_layer_names_model')
    conversion.save_keras_model(model, tfjs_path)

    # Check model.json and weights manifest.
    with open(os.path.join(tfjs_path, 'model.json'), 'rt') as f:
      model_json = json.load(f)

    # Check meta-data in the artifact JSON.
    self.assertEqual(model_json['format'], 'layers-model')
    self.assertEqual(model_json['generatedBy'],
                     'keras v%s' % keras.__version__)
    self.assertEqual(
        model_json['convertedBy'],
        'TensorFlow.js Converter v%s' % version.version)

    self.assertTrue(model_json['modelTopology'])
    weights_manifest = model_json['weightsManifest']
    weight_shapes = dict()
    for group in weights_manifest:
      for weight in group['weights']:
        weight_shapes[weight['name']] = weight['shape']
    self.assertEqual(
        sorted(['dense/kernel', 'dense/bias', 'foo/dense/kernel',
                'foo/dense/bias', 'foo/bar/dense/kernel',
                'foo/bar/dense/bias']),
        sorted(list(weight_shapes.keys())))
    self.assertEqual([12, 2], weight_shapes['dense/kernel'])
    self.assertEqual([2], weight_shapes['dense/bias'])
    self.assertEqual([2, 8], weight_shapes['foo/dense/kernel'])
    self.assertEqual([8], weight_shapes['foo/dense/bias'])
    self.assertEqual([8, 4], weight_shapes['foo/bar/dense/kernel'])
    self.assertEqual([4], weight_shapes['foo/bar/dense/bias'])

  def testConvertMergedModelFromSimpleModelNoSplitByLayer(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MergedDense10')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False,
        kernel_initializer='ones', name='MergedDense20')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModelMerged.h5')
    model.save(h5_path)
    config_json = json.loads(model.to_json(), encoding='utf8')

    # Load the saved weights as a JSON string.
    out, groups = conversion.h5_merged_saved_model_to_tfjs_format(
        h5py.File(h5_path))
    saved_topology = out['model_config']

    # check the model topology was stored
    self.assertEqual(config_json['class_name'], saved_topology['class_name'])
    self.assertEqual(config_json['config'], saved_topology['config'])

    # Check the loaded weights.
    # By default, all weights of the model ought to be put in the same group.
    self.assertEqual(1, len(groups))

    self.assertEqual(keras.__version__, out['keras_version'])
    self.assertEqual('tensorflow', out['backend'])
    weight_group = groups[0]
    self.assertEqual(3, len(weight_group))
    kernel1 = weight_group[0]
    self.assertEqual('MergedDense10/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((3, 4), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 4]), kernel1['data']))
    bias1 = weight_group[1]
    self.assertEqual('MergedDense10/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((4,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([4]), bias1['data']))
    kernel2 = weight_group[2]
    self.assertEqual('MergedDense20/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((4, 2), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([4, 2]), kernel2['data']))

  def testConvertMergedModelFromSimpleModelSplitByLayer(self):
    input_tensor = keras.layers.Input((3,))
    dense1 = keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MergedDense30')(input_tensor)
    output = keras.layers.Dense(
        2, use_bias=False,
        kernel_initializer='ones', name='MergedDense40')(dense1)
    model = keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, 'MyModelMerged.h5')
    model.save(h5_path)
    config_json = json.loads(model.to_json(), encoding='utf8')

    # Load the saved weights as a JSON string.
    out, groups = conversion.h5_merged_saved_model_to_tfjs_format(
        h5py.File(h5_path), split_by_layer=True)
    saved_topology = out['model_config']

    # check the model topology was stored
    self.assertEqual(config_json['class_name'], saved_topology['class_name'])
    self.assertEqual(config_json['config'], saved_topology['config'])

    # Check the loaded weights.
    # Due to `split_by_layer=True`, there ought to be two weight groups,
    # because the model has two layers.
    self.assertEqual(2, len(groups))

    self.assertEqual(keras.__version__, out['keras_version'])
    self.assertEqual('tensorflow', out['backend'])
    self.assertEqual(2, len(groups[0]))
    kernel1 = groups[0][0]
    self.assertEqual('MergedDense30/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((3, 4), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 4]), kernel1['data']))
    bias1 = groups[0][1]
    self.assertEqual('MergedDense30/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((4,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([4]), bias1['data']))
    self.assertEqual(1, len(groups[1]))
    kernel2 = groups[1][0]
    self.assertEqual('MergedDense40/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((4, 2), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([4, 2]), kernel2['data']))

  def testConvertWeightsFromSequentialModelNoSplitByLayer(self):
    sequential_model = keras.models.Sequential([
        keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense10'),
        keras.layers.Dense(
            1, use_bias=False, kernel_initializer='ones', name='Dense20')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    groups = conversion.h5_weights_to_tfjs_format(h5py.File(h5_path))

    # Check the loaded weights.
    # Due to the default `split_by_layer=False`, there should be only one weight
    # group.
    self.assertEqual(1, len(groups))
    self.assertEqual(3, len(groups[0]))
    kernel1 = groups[0][0]
    self.assertEqual('Dense10/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((2, 3), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([2, 3]).tolist(), kernel1['data']))
    bias1 = groups[0][1]
    self.assertEqual('Dense10/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((3,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([3]).tolist(), bias1['data']))
    kernel2 = groups[0][2]
    self.assertEqual('Dense20/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((3, 1), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 1]).tolist(), kernel2['data']))

  def testConvertWeightsFromSequentialModelSplitByLayer(self):
    sequential_model = keras.models.Sequential([
        keras.layers.Dense(
            3, input_shape=(2,), use_bias=True, kernel_initializer='ones',
            name='Dense30'),
        keras.layers.Dense(
            1, use_bias=False, kernel_initializer='ones', name='Dense40')])
    h5_path = os.path.join(self._tmp_dir, 'SequentialModel.h5')
    sequential_model.save_weights(h5_path)

    # Load the saved weights as a JSON string.
    groups = conversion.h5_weights_to_tfjs_format(h5py.File(h5_path),
                                                  split_by_layer=True)

    # Check the loaded weights.
    # Due to the default `split_by_layer=True`, there should be two weight
    # gropus, because the model has two layers.
    self.assertEqual(2, len(groups))
    self.assertEqual(2, len(groups[0]))
    kernel1 = groups[0][0]
    self.assertEqual('Dense30/kernel', kernel1['name'])
    self.assertEqual('float32', kernel1['data'].dtype)
    self.assertEqual((2, 3), kernel1['data'].shape)
    self.assertTrue(np.allclose(np.ones([2, 3]).tolist(), kernel1['data']))
    bias1 = groups[0][1]
    self.assertEqual('Dense30/bias', bias1['name'])
    self.assertEqual('float32', bias1['data'].dtype)
    self.assertEqual((3,), bias1['data'].shape)
    self.assertTrue(np.allclose(np.zeros([3]).tolist(), bias1['data']))

    self.assertEqual(1, len(groups[1]))
    kernel2 = groups[1][0]
    self.assertEqual('Dense40/kernel', kernel2['name'])
    self.assertEqual('float32', kernel2['data'].dtype)
    self.assertEqual((3, 1), kernel2['data'].shape)
    self.assertTrue(np.allclose(np.ones([3, 1]).tolist(), kernel2['data']))

  def testSaveModelSucceedsForNonSequentialModel(self):
    t_input = keras.Input([2])
    dense_layer = keras.layers.Dense(3)
    t_output = dense_layer(t_input)
    model = keras.Model(t_input, t_output)
    conversion.save_keras_model(model, self._tmp_dir)

    # Verify the content of the artifacts output directory.
    self.assertTrue(
        os.path.isfile(os.path.join(self._tmp_dir, 'group1-shard1of1.bin')))
    model_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

    topology_json = model_json['modelTopology']
    self.assertIn('keras_version', topology_json)
    self.assertIn('backend', topology_json)
    self.assertIn('model_config', topology_json)

    weights_manifest = model_json['weightsManifest']
    self.assertIsInstance(weights_manifest, list)
    self.assertEqual(1, len(weights_manifest))
    self.assertIn('paths', weights_manifest[0])

  def testSaveModelSucceedsForTfKerasNonSequentialModel(self):
    t_input = keras.Input([2])
    dense_layer = keras.layers.Dense(3)
    t_output = dense_layer(t_input)
    model = keras.Model(t_input, t_output)

    # `keras.Model`s must be compiled before they can be saved.
    model.compile(loss='mean_squared_error', optimizer='sgd')

    conversion.save_keras_model(model, self._tmp_dir)

    # Verify the content of the artifacts output directory.
    self.assertTrue(
        os.path.isfile(os.path.join(self._tmp_dir, 'group1-shard1of1.bin')))
    model_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

    topology_json = model_json['modelTopology']
    self.assertIn('keras_version', topology_json)
    self.assertIn('backend', topology_json)
    self.assertIn('model_config', topology_json)

    weights_manifest = model_json['weightsManifest']
    self.assertIsInstance(weights_manifest, list)
    self.assertEqual(1, len(weights_manifest))
    self.assertIn('paths', weights_manifest[0])

  def testSaveModelSucceedsForNestedKerasModel(self):
    inner_model = keras.Sequential([
        keras.layers.Dense(4, input_shape=[3], activation='relu'),
        keras.layers.Dense(3, activation='tanh')])
    outer_model = keras.Sequential()
    outer_model.add(inner_model)
    outer_model.add(keras.layers.Dense(1, activation='sigmoid'))

    conversion.save_keras_model(outer_model, self._tmp_dir)

    # Verify the content of the artifacts output directory.
    self.assertTrue(
        os.path.isfile(os.path.join(self._tmp_dir, 'group1-shard1of1.bin')))
    model_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

    topology_json = model_json['modelTopology']
    self.assertIn('keras_version', topology_json)
    self.assertIn('backend', topology_json)
    self.assertIn('model_config', topology_json)

    # Verify that all the layers' weights are present.
    weights_manifest = model_json['weightsManifest']
    self.assertIsInstance(weights_manifest, list)
    weight_entries = []
    for group in weights_manifest:
      weight_entries.extend(group['weights'])
    self.assertEqual(6, len(weight_entries))

  def testSaveModelSucceedsForTfKerasSequentialModel(self):
    model = keras.Sequential([keras.layers.Dense(1, input_shape=[2])])

    # `keras.Model`s must be compiled before they can be saved.
    model.compile(loss='mean_squared_error', optimizer='sgd')

    conversion.save_keras_model(model, self._tmp_dir)

    # Verify the content of the artifacts output directory.
    self.assertTrue(
        os.path.isfile(os.path.join(self._tmp_dir, 'group1-shard1of1.bin')))
    model_json = json.load(
        open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

    topology_json = model_json['modelTopology']
    self.assertIn('keras_version', topology_json)
    self.assertIn('backend', topology_json)
    self.assertIn('model_config', topology_json)

    weights_manifest = model_json['weightsManifest']
    self.assertIsInstance(weights_manifest, list)
    self.assertEqual(1, len(weights_manifest))
    self.assertIn('paths', weights_manifest[0])

  def testSavedModelSucceedsForExistingDirAndSequential(self):
    artifacts_dir = os.path.join(self._tmp_dir, 'artifacts')
    os.makedirs(artifacts_dir)
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, input_shape=[2]))
    conversion.save_keras_model(model, artifacts_dir)

    # Verify the content of the artifacts output directory.
    self.assertTrue(
        os.path.isfile(os.path.join(artifacts_dir, 'group1-shard1of1.bin')))
    model_json = json.load(
        open(os.path.join(artifacts_dir, 'model.json'), 'rt'))

    topology_json = model_json['modelTopology']
    self.assertIn('keras_version', topology_json)
    self.assertIn('backend', topology_json)
    self.assertIn('model_config', topology_json)

    weights_manifest = model_json['weightsManifest']
    self.assertIsInstance(weights_manifest, list)
    self.assertEqual(1, len(weights_manifest))
    self.assertIn('paths', weights_manifest[0])

  def testSavedModelRaisesErrorIfArtifactsDirExistsAsAFile(self):
    artifacts_dir = os.path.join(self._tmp_dir, 'artifacts')
    with open(artifacts_dir, 'wt') as f:
      f.write('foo\n')
    t_input = keras.Input([2])
    dense_layer = keras.layers.Dense(3)
    t_output = dense_layer(t_input)
    model = keras.Model(t_input, t_output)
    with self.assertRaisesRegexp(  # pylint: disable=deprecated-method
        ValueError, r'already exists as a file'):
      conversion.save_keras_model(model, artifacts_dir)

  def testTranslateBatchNormalizationV1ClassName(self):
    # The config JSON of a model consisting of a "BatchNormalizationV1" layer.
    # pylint: disable=line-too-long
    json_object = json.loads(
        '{"class_name": "Sequential", "keras_version": "2.2.4-tf", "config": {"layers": [{"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "name": "dense", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "relu", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "units": 10, "batch_input_shape": [null, 3], "use_bias": true, "activity_regularizer": null}}, {"class_name": "BatchNormalizationV1", "config": {"beta_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "name": "batch_normalization_v1", "dtype": "float32", "trainable": true, "moving_variance_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "scale": true, "axis": [1], "epsilon": 0.001, "gamma_constraint": null, "gamma_regularizer": null, "beta_regularizer": null, "momentum": 0.99, "center": true}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "name": "dense_1", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "units": 1, "use_bias": true, "activity_regularizer": null}}], "name": "sequential"}, "backend": "tensorflow"}')
    # pylint: enable=line-too-long
    conversion.translate_class_names(json_object)
    # Some class names should not have been changed be translate_class_names().
    self.assertEqual(json_object['class_name'], 'Sequential')
    self.assertEqual(json_object['keras_version'], '2.2.4-tf')
    self.assertEqual(json_object['config']['layers'][0]['class_name'], 'Dense')
    # The translation should have happend:
    #     BatchNormalizationV1 --> BatchNormalization.
    self.assertEqual(
        json_object['config']['layers'][1]['class_name'], 'BatchNormalization')
    self.assertEqual(json_object['config']['layers'][2]['class_name'], 'Dense')

    # Assert that converted JSON can be reconstituted as a model object.
    model = keras.models.model_from_json(json.dumps(json_object))
    self.assertIsInstance(model, keras.Sequential)
    self.assertEqual(model.input_shape, (None, 3))
    self.assertEqual(model.output_shape, (None, 1))
    self.assertEqual(model.layers[0].units, 10)
    self.assertEqual(model.layers[2].units, 1)

  def testTranslateUnifiedGRUAndLSTMClassName(self):
    # The config JSON of a model consisting of a "UnifiedGRU" layer
    # and a "UnifiedLSTM" layer.
    # pylint: disable=line-too-long
    json_object = json.loads(
        '{"class_name": "Sequential", "keras_version": "2.2.4-tf", "config": {"layers": [{"class_name": "UnifiedGRU", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "tanh", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "batch_input_shape": [null, 4, 3], "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "reset_after": true, "recurrent_regularizer": null, "name": "unified_gru", "bias_constraint": null, "go_backwards": false, "implementation": 1, "kernel_regularizer": null, "return_sequences": true, "recurrent_constraint": null}}, {"class_name": "UnifiedLSTM", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "tanh", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 2, "unit_forget_bias": true, "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "recurrent_regularizer": null, "name": "unified_lstm", "bias_constraint": null, "go_backwards": false, "implementation": 1, "kernel_regularizer": null, "return_sequences": false, "recurrent_constraint": null}}], "name": "sequential"}, "backend": "tensorflow"}')
    # pylint: enable=line-too-long
    conversion.translate_class_names(json_object)
    # Some class names should not have been changed be translate_class_names().
    self.assertEqual(json_object['class_name'], 'Sequential')
    self.assertEqual(json_object['keras_version'], '2.2.4-tf')
    # The translation should have happend:
    #     UnifiedGRU --> GRU.
    #     UnifiedLSTM --> LSTM.
    self.assertEqual(json_object['config']['layers'][0]['class_name'], 'GRU')
    self.assertEqual(json_object['config']['layers'][1]['class_name'], 'LSTM')

    # Assert that converted JSON can be reconstituted as a model object.
    model = keras.models.model_from_json(json.dumps(json_object))
    self.assertIsInstance(model, keras.Sequential)
    self.assertEqual(model.input_shape, (None, 4, 3))
    self.assertEqual(model.output_shape, (None, 2))


if __name__ == '__main__':
  tf.test.main()
