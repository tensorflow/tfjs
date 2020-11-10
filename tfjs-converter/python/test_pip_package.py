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
"""Test the Python API and shell binary of the tensorflowjs pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v1 import saved_model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training.tracking import tracking
from tensorflow.python.saved_model.save import save
import tensorflow_hub as hub

import tensorflowjs as tfjs


def _createKerasModel(layer_name_prefix, h5_path=None):
  """Create a Keras model for testing.

  Args:
    layer_name_prefix: A prefix string for layer names. This helps avoid
      clashes in layer names between different test methods.
    h5_path: Optional string path for a HDF5 (.h5) file to save the model
      in.

  Returns:
    An instance of tf.keras.Model.
  """
  input_tensor = tf.keras.layers.Input((3, ))
  dense1 = tf.keras.layers.Dense(
      4,
      use_bias=True,
      kernel_initializer='ones',
      bias_initializer='zeros',
      name=layer_name_prefix + '1')(input_tensor)
  output = tf.keras.layers.Dense(
      2,
      use_bias=False,
      kernel_initializer='ones',
      name=layer_name_prefix + '2')(dense1)
  model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output])
  model.compile(optimizer='adam', loss='binary_crossentropy')
  model.predict(tf.ones((1, 3)), steps=1)

  if h5_path:
    model.save(h5_path, save_format='h5')
  return model

def _createTensorFlowSavedModelV1(name_scope, save_path):
  """Create a TensorFlow SavedModel for testing.
  Args:
    name_scope: Name scope to create the model under. This helps avoid
      op and variable name clashes between different test methods.
    save_path: The directory path in which to save the model.
  """
  graph = tf.Graph()
  with graph.as_default():
    with tf.compat.v1.name_scope(name_scope):
      x = tf.compat.v1.constant([[37.0, -23.0], [1.0, 4.0]])
      w = tf.compat.v1.get_variable('w', shape=[2, 2])
      y = tf.compat.v1.matmul(x, w)
      output = tf.compat.v1.nn.softmax(y)
      init_op = w.initializer

      # Create a builder.
      builder = saved_model.builder.SavedModelBuilder(save_path)

      with tf.compat.v1.Session() as sess:
        # Run the initializer on `w`.
        sess.run(init_op)

        builder.add_meta_graph_and_variables(
            sess, [saved_model.tag_constants.SERVING],
            signature_def_map={
                "serving_default":
                    saved_model.signature_def_utils.predict_signature_def(
                        inputs={"x": x},
                        outputs={"output": output})
            },
            assets_collection=None)

      builder.save()

def _createTensorFlowSavedModel(save_path):
  """Create a TensorFlow SavedModel for testing.

  Args:
    save_path: The directory path in which to save the model.
  """

  input_data = constant_op.constant(1., shape=[1])
  root = tracking.AutoTrackable()
  root.v1 = variables.Variable(3.)
  root.v2 = variables.Variable(2.)
  root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
  to_save = root.f.get_concrete_function(input_data)

  save(root, save_path, to_save)


def _create_hub_module(save_path):
  """Create a TensorFlow Hub module for testing.

  Args:
    save_path: The directory path in which to save the model.
  """
  # Module function that doubles its input.
  def double_module_fn():
    w = tf.Variable([2.0, 4.0])
    x = tf.compat.v1.placeholder(dtype=tf.float32)
    hub.add_signature(inputs=x, outputs=x*w)
  graph = tf.Graph()
  with graph.as_default():
    spec = hub.create_module_spec(double_module_fn)
    m = hub.Module(spec)
  # Export the module.
  with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    m.export(save_path, sess)

def _create_frozen_model(save_path):
  graph = tf.Graph()
  saved_model_dir = os.path.join(save_path)
  with graph.as_default():
    x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
    w = tf.Variable(tf.random.uniform([2, 2]))
    y = tf.matmul(x, w)
    tf.nn.softmax(y)
    init_op = w.initializer

    # Create a builder
    builder = saved_model.builder.SavedModelBuilder(
        saved_model_dir)

    with tf.compat.v1.Session() as sess:
      # Run the initializer on `w`.
      sess.run(init_op)

      builder.add_meta_graph_and_variables(
          sess, [saved_model.tag_constants.SERVING],
          signature_def_map=None,
          assets_collection=None)

    builder.save()

  frozen_file = os.path.join(save_path, 'frozen.pb')
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
      saved_model_tags=saved_model.tag_constants.SERVING,
      input_saved_model_dir=saved_model_dir)
class APIAndShellTest(tf.test.TestCase):
  """Tests for the Python API of the pip package."""

  @classmethod
  def setUpClass(cls):
    cls.class_tmp_dir = tempfile.mkdtemp()
    cls.tf_saved_model_dir = os.path.join(cls.class_tmp_dir, 'tf_saved_model')
    cls.tf_saved_model_v1_dir = os.path.join(
                cls.class_tmp_dir, 'tf_saved_model_v1')
    cls.tf_frozen_model_dir = os.path.join(cls.class_tmp_dir, 'tf_frozen_model')
    _createTensorFlowSavedModel(cls.tf_saved_model_dir)
    _createTensorFlowSavedModelV1('b', cls.tf_saved_model_v1_dir)
    _create_frozen_model(cls.tf_frozen_model_dir)
    cls.tf_hub_module_dir = os.path.join(cls.class_tmp_dir, 'tf_hub_module')
    _create_hub_module(cls.tf_hub_module_dir)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.class_tmp_dir)

  def setUp(self):
    # Make sure this file is not being run from the source directory, to
    # avoid picking up source files.
    if os.path.isdir(
        os.path.join(os.path.dirname(__file__), 'tensorflowjs')):
      self.fail('Do not run this test from the Python source directory. '
                'This file is intended to be run on pip install.')

    self._tmp_dir = tempfile.mkdtemp()
    super(APIAndShellTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(APIAndShellTest, self).tearDown()

  def testVersionString(self):
    self.assertEqual(2, tfjs.__version__.count('.'))

  def testSaveKerasModel(self):
    with self.test_session():
      # First create a toy keras model.
      model = _createKerasModel('MergedDense')

      tfjs.converters.save_keras_model(model, self._tmp_dir)

      # Briefly check the model topology.
      with open(os.path.join(self._tmp_dir, 'model.json')) as f:
        json_content = json.load(f)
      model_json = json_content['modelTopology']
      self.assertIsInstance(model_json['model_config'], dict)
      self.assertIsInstance(model_json['model_config']['config'], dict)
      self.assertIn('layers', model_json['model_config']['config'])

      weights_manifest = json_content['weightsManifest']
      self.assertIsInstance(weights_manifest, list)

      # Briefly check the weights manifest.
      weight_shapes = dict()
      weight_dtypes = dict()
      for manifest_item in weights_manifest:
        for weight in manifest_item['weights']:
          weight_name = weight['name']
          weight_shapes[weight_name] = weight['shape']
          weight_dtypes[weight_name] = weight['dtype']

      self.assertEqual(
          sorted(list(weight_shapes.keys())),
          sorted([
              'MergedDense1/kernel', 'MergedDense1/bias',
              'MergedDense2/kernel'
          ]))
      self.assertEqual(weight_shapes['MergedDense1/kernel'], [3, 4])
      self.assertEqual(weight_shapes['MergedDense1/bias'], [4])
      self.assertEqual(weight_shapes['MergedDense2/kernel'], [4, 2])
      self.assertEqual(weight_dtypes['MergedDense1/kernel'], 'float32')
      self.assertEqual(weight_dtypes['MergedDense1/bias'], 'float32')
      self.assertEqual(weight_dtypes['MergedDense2/kernel'], 'float32')

  def testLoadKerasModel(self):
    # Use separate tf.Graph and tf.compat.v1.Session contexts
    # to prevent name collision.
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # First create a toy keras model.
      model1 = _createKerasModel('MergedDense')
      tfjs.converters.save_keras_model(model1, self._tmp_dir)
      model1_weight_values = model1.get_weights()

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # Load the model from saved artifacts.
      model2 = tfjs.converters.load_keras_model(
          os.path.join(self._tmp_dir, 'model.json'))

      # Compare the loaded model with the original one.
      model2_weight_values = model2.get_weights()
      self.assertEqual(len(model1_weight_values), len(model2_weight_values))
      for model1_weight_value, model2_weight_value in zip(
          model1_weight_values, model2_weight_values):
        self.assertAllClose(model1_weight_value, model2_weight_value)

    # Check the content of the output directory.
    self.assertTrue(glob.glob(os.path.join(self._tmp_dir, 'group*-*')))

  def testInvalidInputFormatRaisesError(self):
    process = subprocess.Popen(
        [
            'tensorflowjs_converter', '--input_format',
            'nonsensical_format', self._tmp_dir, self._tmp_dir
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(b'--input_format', tf.compat.as_bytes(stderr))

  def testMissingInputPathRaisesError(self):
    process = subprocess.Popen(
        [
            'tensorflowjs_converter'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(b'input_path', tf.compat.as_bytes(stderr))

  def testKerasH5ConversionWorksFromCLI(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # First create a toy keras model.
      os.makedirs(os.path.join(self._tmp_dir, 'keras_h5'))
      h5_path = os.path.join(self._tmp_dir, 'keras_h5', 'model.h5')
      _createKerasModel('MergedDenseForCLI', h5_path)

      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras', h5_path,
          self._tmp_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # Briefly check the model topology.
      with open(os.path.join(self._tmp_dir, 'model.json'), 'rt') as f:
        json_content = json.load(f)
      model_json = json_content['modelTopology']
      self.assertIsInstance(model_json['model_config'], dict)
      self.assertIsInstance(model_json['model_config']['config'], dict)
      self.assertIn('layers', model_json['model_config']['config'])

      weights_manifest = json_content['weightsManifest']
      self.assertIsInstance(weights_manifest, list)

      # Briefly check the weights manifest.
      weight_shapes = dict()
      weight_dtypes = dict()
      for manifest_item in weights_manifest:
        for weight in manifest_item['weights']:
          weight_name = weight['name']
          weight_shapes[weight_name] = weight['shape']
          weight_dtypes[weight_name] = weight['dtype']

      self.assertEqual(
          sorted(list(weight_shapes.keys())),
          sorted([
              'MergedDenseForCLI1/kernel', 'MergedDenseForCLI1/bias',
              'MergedDenseForCLI2/kernel'
          ]))
      self.assertEqual(weight_shapes['MergedDenseForCLI1/kernel'], [3, 4])
      self.assertEqual(weight_shapes['MergedDenseForCLI1/bias'], [4])
      self.assertEqual(weight_shapes['MergedDenseForCLI2/kernel'], [4, 2])
      self.assertEqual(weight_dtypes['MergedDenseForCLI1/kernel'], 'float32')
      self.assertEqual(weight_dtypes['MergedDenseForCLI1/bias'], 'float32')
      self.assertEqual(weight_dtypes['MergedDenseForCLI2/kernel'], 'float32')

      # Verify that there is only one weight group due to the default
      # non-split_weights_by_layer behavior. The model is a small one, which
      # does not exceed the 4-MB shard size limit. Therefore, there should
      # be only one weight file.
      self.assertEqual(
          1, len(glob.glob(os.path.join(self._tmp_dir, 'group*'))))

  def testKerasH5ConversionSplitWeightsByLayerWorksFromCLI(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # First create a toy keras model.
      os.makedirs(os.path.join(self._tmp_dir, 'keras_h5'))
      h5_path = os.path.join(self._tmp_dir, 'keras_h5', 'model.h5')
      _createKerasModel('MergedDenseForCLI', h5_path)

      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras',
          '--split_weights_by_layer', h5_path, self._tmp_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # Briefly check the model topology.
      with open(os.path.join(self._tmp_dir, 'model.json'), 'rt') as f:
        json_content = json.load(f)
      model_json = json_content['modelTopology']
      self.assertIsInstance(model_json['model_config'], dict)
      self.assertIsInstance(model_json['model_config']['config'], dict)
      self.assertIn('layers', model_json['model_config']['config'])

      weights_manifest = json_content['weightsManifest']
      self.assertIsInstance(weights_manifest, list)

      # Briefly check the weights manifest.
      weight_shapes = dict()
      weight_dtypes = dict()
      for manifest_item in weights_manifest:
        for weight in manifest_item['weights']:
          weight_name = weight['name']
          weight_shapes[weight_name] = weight['shape']
          weight_dtypes[weight_name] = weight['dtype']

      self.assertEqual(
          sorted(list(weight_shapes.keys())),
          sorted([
              'MergedDenseForCLI1/kernel', 'MergedDenseForCLI1/bias',
              'MergedDenseForCLI2/kernel'
          ]))
      self.assertEqual(weight_shapes['MergedDenseForCLI1/kernel'], [3, 4])
      self.assertEqual(weight_shapes['MergedDenseForCLI1/bias'], [4])
      self.assertEqual(weight_shapes['MergedDenseForCLI2/kernel'], [4, 2])
      self.assertEqual(weight_dtypes['MergedDenseForCLI1/kernel'], 'float32')
      self.assertEqual(weight_dtypes['MergedDenseForCLI1/bias'], 'float32')
      self.assertEqual(weight_dtypes['MergedDenseForCLI2/kernel'], 'float32')

      # Verify that there are two weight groups due to the optional flag
      # --split_weights_by_layer behavior. The model is a small one. None of
      # the layers should have weight sizes exceeding the 4-MB shard size
      # limit.
      self.assertEqual(
          2, len(glob.glob(os.path.join(self._tmp_dir, 'group*'))))

  def testKerasH5ConversionWithSignatureNameErrors(self):
    process = subprocess.Popen(
        [
            'tensorflowjs_converter', '--input_format', 'keras',
            '--signature_name', 'bar',
            os.path.join(self._tmp_dir, 'foo.h5'),
            os.path.join(self._tmp_dir, 'output')
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    self.assertGreater(process.returncode, 0)
    self.assertIn(
        b'The --signature_name flag is applicable only to',
        tf.compat.as_bytes(stderr))

  def testConvertTFSavedModelV1WithCommandLineWorks(self):
    output_dir = os.path.join(self._tmp_dir)
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        self.tf_saved_model_v1_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weights = [{
        'paths': ['group1-shard1of1.bin'],
        'weights': [{'dtype': 'float32', 'name': 'w', 'shape': [2, 2]}]}]

    # Load the saved weights as a JSON string.
    output_json = json.load(
        open(os.path.join(output_dir, 'model.json'), 'rt'))
    self.assertEqual(output_json['weightsManifest'], weights)

    # Check the content of the output directory.
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))


  def testConvertTFHubModuleWithCommandLineWorks(self):
    output_dir = os.path.join(self._tmp_dir)
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_hub',
        self.tf_hub_module_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weights = [{
        'paths': ['group1-shard1of1.bin'],
        'weights': [{
            'shape': [2],
            'name': 'module/Variable',
            'dtype': 'float32'
        }]
    }]
    # Load the saved weights as a JSON string.
    output_json = json.load(
        open(os.path.join(output_dir, 'model.json'), 'rt'))
    self.assertEqual(output_json['weightsManifest'], weights)

    # Check the content of the output directory.
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testConvertTFSavedModelWithCommandLineWorks(self):
    output_dir = os.path.join(self._tmp_dir)
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        self.tf_saved_model_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weights = [{
        'paths': ['group1-shard1of1.bin'],
        'weights': [{
            'dtype': 'float32',
            'shape': [],
            'name': 'StatefulPartitionedCall/mul'
        }]
    }]

    # Load the saved weights as a JSON string.
    output_json = json.load(
        open(os.path.join(output_dir, 'model.json'), 'rt'))
    weights_manifest = output_json['weightsManifest']
    self.assertEqual(len(weights_manifest), len(weights))
    if sys.version_info[0] < 3:
      self.assertItemsEqual(weights_manifest[0]['paths'],
                            weights[0]['paths'])
      self.assertItemsEqual(weights_manifest[0]['weights'],
                            weights[0]['weights'])
    else:
      self.assertCountEqual(weights_manifest[0]['paths'],
                            weights[0]['paths'])
      self.assertCountEqual(weights_manifest[0]['weights'],
                            weights[0]['weights'])

    # Check the content of the output directory.
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testConvertTFSavedModelIntoShardedWeights(self):
    output_dir = os.path.join(self._tmp_dir, 'tfjs_model')
    # Do initial conversion without sharding.
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        self.tf_saved_model_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weight_files = glob.glob(os.path.join(output_dir, 'group*.bin'))

    # Get size of weights in bytes after graph optimizations.
    optimized_total_weight = sum([os.path.getsize(f) for f in weight_files])
    # Due to the shard size, there ought to be 2 shards after conversion.
    weight_shard_size_bytes = int(optimized_total_weight * 0.8)

    output_dir = os.path.join(self._tmp_dir, 'sharded_model')
    # Convert Saved Model again with shard argument set.
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        '--weight_shard_size_bytes', str(weight_shard_size_bytes),
        self.tf_saved_model_dir, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    weight_files = sorted(glob.glob(os.path.join(output_dir, 'group*.bin')))
    self.assertEqual(len(weight_files), 2)
    weight_file_sizes = [os.path.getsize(f) for f in weight_files]
    self.assertEqual(sum(weight_file_sizes), optimized_total_weight)
    self.assertLess(weight_file_sizes[1], weight_file_sizes[0])

  def testConvertTFFrozenModelWithCommandLineWorks(self):
    output_dir = os.path.join(self._tmp_dir)
    frozen_file = os.path.join(self.tf_frozen_model_dir, 'frozen.pb')
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tf_frozen_model',
        '--output_format', 'tfjs_graph_model', '--output_node_names',
        'Softmax',
        frozen_file, output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # Check model.json and weights manifest.
    with open(os.path.join(output_dir, 'model.json'), 'rt') as f:
      model_json = json.load(f)
    self.assertTrue(model_json['modelTopology'])
    self.assertIsNot(model_json['modelTopology']['versions'], None)
    signature = model_json['signature']
    self.assertIsNot(signature, None)
    # frozen model signature has no inputs
    self.assertIsNot(signature['outputs'], None)

    weights_manifest = model_json['weightsManifest']
    weights_manifest = model_json['weightsManifest']
    self.assertCountEqual(weights_manifest[0]['paths'],
                          ['group1-shard1of1.bin'])
    self.assertIn('weights', weights_manifest[0])
    self.assertTrue(
        glob.glob(
            os.path.join(self._tmp_dir, 'group*-*')))

    # Check the content of the output directory.
    self.assertTrue(glob.glob(os.path.join(output_dir, 'group*-*')))

  def testConvertTensorflowjsArtifactsToKerasH5(self):
    # 1. Create a toy keras model and save it as an HDF5 file.
    os.makedirs(os.path.join(self._tmp_dir, 'keras_h5'))
    h5_path = os.path.join(self._tmp_dir, 'keras_h5', 'model.h5')
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = _createKerasModel('MergedDenseForCLI', h5_path)
      model_json = model.to_json()

    # 2. Convert the HDF5 file to tensorflowjs format.
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'keras', h5_path,
        self._tmp_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 3. Convert the tensorflowjs artifacts back to HDF5.
    new_h5_path = os.path.join(self._tmp_dir, 'model_2.h5')
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
        '--output_format', 'keras',
        os.path.join(self._tmp_dir, 'model.json'), new_h5_path])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 4. Load the model back from the new HDF5 file and compare with the
    #    original model.
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model_2 = tf.keras.models.load_model(new_h5_path)
      model_2_json = model_2.to_json()
      self.assertEqual(model_json, model_2_json)

  def testLoadTensorflowjsArtifactsAsKerasModel(self):
    # 1. Create a toy keras model and save it as an HDF5 file.
    os.makedirs(os.path.join(self._tmp_dir, 'keras_h5'))
    h5_path = os.path.join(self._tmp_dir, 'keras_h5', 'model.h5')
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model = _createKerasModel('MergedDenseForCLI', h5_path)
      model_json = model.to_json()

    # 2. Convert the HDF5 file to tensorflowjs format.
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'keras', h5_path,
        self._tmp_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 3. Load the tensorflowjs artifacts as a tf.keras.Model instance.
    with tf.Graph().as_default(), tf.compat.v1.Session():
      model_2 = tfjs.converters.load_keras_model(
          os.path.join(self._tmp_dir, 'model.json'))
      model_2_json = model_2.to_json()
      self.assertEqual(model_json, model_2_json)

  def testVersion(self):
    process = subprocess.Popen(
        ['tensorflowjs_converter', '--version'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    self.assertEqual(0, process.returncode)
    self.assertIn(
        tf.compat.as_bytes('tensorflowjs %s' % tfjs.__version__),
        tf.compat.as_bytes(stdout))

    process = subprocess.Popen(
        ['tensorflowjs_converter', '-v'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    self.assertEqual(0, process.returncode)
    self.assertIn(
        tf.compat.as_bytes('tensorflowjs %s' % tfjs.__version__),
        tf.compat.as_bytes(stdout))


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

  def testConvertTfKerasNestedSequentialSavedModelIntoTfjsFormat(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      x = np.random.randn(8, 10)

      # 1. Run the model.predict(), store the result. Then saved the model
      #    as a SavedModel.
      model = self._createNestedSequentialModel()
      y = model.predict(x)

      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Implicit value of --output_format: tfjs_layers_model
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras_saved_model',
          self._tmp_dir, tfjs_output_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      self.assertTrue(os.path.isfile(model_json_path))

      # 3. Convert the tfjs model to keras h5 format.
      new_h5_path = os.path.join(self._tmp_dir, 'new_h5.h5')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          '--output_format', 'keras', model_json_path, new_h5_path])
      process.communicate()
      self.assertEqual(0, process.returncode)

      self.assertTrue(os.path.isfile(new_h5_path))

      # 4. Load the model back and assert on the equality of the predict
      #    results.
      model_prime = tf.keras.models.load_model(new_h5_path)
      new_y = model_prime.predict(x)
      self.assertAllClose(y, new_y)

  def testConvertTfKerasFunctionalSavedModelIntoTfjsFormat(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      x1 = np.random.randn(4, 8)
      x2 = np.random.randn(4, 10)

      # 1. Run the model.predict(), store the result. Then saved the model
      #    as a SavedModel.
      model = self._createFunctionalModelWithWeights()
      y = model.predict([x1, x2])

      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Use explicit --output_format value: tfjs_layers_model
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras_saved_model',
          '--output_format', 'tfjs_layers_model',
          self._tmp_dir, tfjs_output_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      model_json_path = os.path.join(tfjs_output_dir, 'model.json')
      self.assertTrue(os.path.isfile(model_json_path))

      # 3. Convert the tfjs model to keras h5 format.
      new_h5_path = os.path.join(self._tmp_dir, 'new_h5.h5')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          '--output_format', 'keras', model_json_path, new_h5_path])
      process.communicate()
      self.assertEqual(0, process.returncode)

      self.assertTrue(os.path.isfile(new_h5_path))

      # 4. Load the model back and assert on the equality of the predict
      #    results.
      model_prime = tf.keras.models.load_model(new_h5_path)
      new_y = model_prime.predict([x1, x2])
      self.assertAllClose(y, new_y)

  def testUsingIncorrectKerasSavedModelRaisesError(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 1. Run the model.predict(), store the result. Then saved the model
      #    as a SavedModel.
      model = self._createNestedSequentialModel()
      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Use incorrect --input_format value: keras
      process = subprocess.Popen(
          [
              'tensorflowjs_converter', '--input_format', 'keras',
              self._tmp_dir, tfjs_output_dir
          ],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
      _, stderr = process.communicate()
      self.assertIn(
          b'Expected path to point to an HDF5 file, '
          b'but it points to a directory', tf.compat.as_bytes(stderr))

  def testConvertTfjsLayersModelIntoShardedWeights(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      x = np.random.randn(8, 10)

      # 1. Run the model.predict(), store the result. Then saved the model
      #    as a SavedModel.
      model = self._createNestedSequentialModel()
      y = model.predict(x)

      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs_layers_model format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Implicit value of --output_format: tfjs_layers_model
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras_saved_model',
          self._tmp_dir, tfjs_output_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 3. Convert the tfjs_layers_model to another tfjs_layers_model,
      #    with sharded weights.
      weight_shard_size_bytes = int(total_weight_bytes * 0.3)
      # Due to the shard size, there ought to be 4 shards after conversion.
      sharded_model_dir = os.path.join(self._tmp_dir, 'tfjs_sharded')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          '--output_format', 'tfjs_layers_model',
          '--weight_shard_size_bytes', str(weight_shard_size_bytes),
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 4. Check the sharded weight files and their sizes.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_dir, 'group*.bin')))
      self.assertEqual(len(weight_files), 4)
      weight_file_sizes = [os.path.getsize(f) for f in weight_files]
      self.assertEqual(sum(weight_file_sizes), total_weight_bytes)
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[1])
      self.assertEqual(weight_file_sizes[0], weight_file_sizes[2])
      self.assertLess(weight_file_sizes[3], weight_file_sizes[0])

      # 5. Convert the sharded tfjs_layers_model back into a keras h5 file.
      new_h5_path = os.path.join(self._tmp_dir, 'new_h5.h5')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          os.path.join(sharded_model_dir, 'model.json'), new_h5_path
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 6. Load the keras model and check the predict() output is close to
      #    before.
      new_model = tf.keras.models.load_model(new_h5_path)
      new_y = new_model.predict(x)
      self.assertAllClose(new_y, y)

  def testConvertTfjsLayersModelWithLegacyQuantization(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 1. Saved the model as a SavedModel.
      model = self._createNestedSequentialModel()

      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs_layers_model format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Implicit value of --output_format: tfjs_layers_model
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras_saved_model',
          self._tmp_dir, tfjs_output_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 3. Convert the tfjs_layers_model to another tfjs_layers_model,
      #    with uint16 quantization.
      sharded_model_dir = os.path.join(self._tmp_dir, 'tfjs_sharded')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          '--output_format', 'tfjs_layers_model',
          '--quantization_bytes', '2',
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 4. Check the quantized weight file and its size.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_dir, 'group*.bin')))
      self.assertEqual(len(weight_files), 1)
      weight_file_size = os.path.getsize(weight_files[0])
      # The size of the weight file should reflect the uint16 quantization.
      self.assertEqual(weight_file_size, total_weight_bytes // 2)


  def testConvertTfjsLayersModelWithQuantization(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 1. Saved the model as a SavedModel.
      model = self._createNestedSequentialModel()

      weights = model.get_weights()
      total_weight_bytes = sum(np.size(w) for w in weights) * 4

      tf.keras.models.save_model(model, self._tmp_dir)

      # 2. Convert the keras saved model to tfjs_layers_model format.
      tfjs_output_dir = os.path.join(self._tmp_dir, 'tfjs')
      # Implicit value of --output_format: tfjs_layers_model
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'keras_saved_model',
          self._tmp_dir, tfjs_output_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 3. Convert the tfjs_layers_model to another tfjs_layers_model,
      #    with uint16 quantization.
      sharded_model_dir = os.path.join(self._tmp_dir, 'tfjs_sharded')
      process = subprocess.Popen([
          'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
          '--output_format', 'tfjs_layers_model',
          '--quantize_uint16', '*',
          os.path.join(tfjs_output_dir, 'model.json'), sharded_model_dir
      ])
      process.communicate()
      self.assertEqual(0, process.returncode)

      # 4. Check the quantized weight file and its size.
      weight_files = sorted(
          glob.glob(os.path.join(sharded_model_dir, 'group*.bin')))
      self.assertEqual(len(weight_files), 1)
      weight_file_size = os.path.getsize(weight_files[0])
      # The size of the weight file should reflect the uint16 quantization.
      self.assertEqual(weight_file_size, total_weight_bytes // 2)

  def testConvertTfjsLayersModelToTfjsGraphModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 1. Create a model for testing.
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=[4]))
      model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
      model.compile(optimizer='adam', loss='binary_crossentropy')
      model.predict(tf.ones((1, 4)), steps=1)

      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path, save_format='h5')

    # 2. Convert the keras saved model to tfjs_layers_model format.
    layers_model_output_dir = os.path.join(self._tmp_dir, 'tfjs_layers')
    # Implicit value of --output_format: tfjs_layers_model
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'keras',
        h5_path, layers_model_output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 3. Convert the tfjs_layers_model to another tfjs_graph_model.
    graph_model_dir = os.path.join(self._tmp_dir, 'tfjs_graph')
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
        '--output_format', 'tfjs_graph_model',
        os.path.join(layers_model_output_dir, 'model.json'), graph_model_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 4. Check the model.json and weight file and its size.
    self.assertTrue(os.path.isfile(os.path.join(graph_model_dir, 'model.json')))
    weight_files = sorted(
        glob.glob(os.path.join(graph_model_dir, 'group*.bin')))
    self.assertEqual(len(weight_files), 1)

  def testConvertTfjsLayersModelToKerasSavedModel(self):
    with tf.Graph().as_default(), tf.compat.v1.Session():
      # 1. Create a model for testing.
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=[4]))
      model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
      model.compile(optimizer='adam', loss='binary_crossentropy')
      model.predict(tf.ones((1, 4)), steps=1)

      h5_path = os.path.join(self._tmp_dir, 'model.h5')
      model.save(h5_path, save_format='h5')

    # 2. Convert the keras saved model to tfjs_layers_model format.
    layers_model_output_dir = os.path.join(self._tmp_dir, 'tfjs_layers')
    # Implicit value of --output_format: tfjs_layers_model
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'keras',
        h5_path, layers_model_output_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 3. Convert the tfjs_layers_model to another keras_saved_model.
    keras_saved_model_dir = os.path.join(self._tmp_dir, 'keras_saved_model')
    process = subprocess.Popen([
        'tensorflowjs_converter', '--input_format', 'tfjs_layers_model',
        '--output_format', 'keras_saved_model',
        os.path.join(layers_model_output_dir, 'model.json'),
        keras_saved_model_dir
    ])
    process.communicate()
    self.assertEqual(0, process.returncode)

    # 4. Check the files that belong to the conversion result.
    files = glob.glob(os.path.join(keras_saved_model_dir, '*'))
    self.assertIn(os.path.join(keras_saved_model_dir, 'saved_model.pb'), files)
    self.assertIn(os.path.join(keras_saved_model_dir, 'variables'), files)
    self.assertIn(os.path.join(keras_saved_model_dir, 'assets'), files)


if __name__ == '__main__':
  tf.test.main()
