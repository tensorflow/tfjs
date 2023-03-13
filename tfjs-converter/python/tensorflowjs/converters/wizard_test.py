# Copyright 2019 Google LLC
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import tempfile
import json
import os
import shutil
import tensorflow.compat.v2 as tf
from tensorflow.python.eager import def_function
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.saved_model import save

from tensorflowjs.converters import wizard

SAVED_MODEL_DIR = 'saved_model'
SAVED_MODEL_NAME = 'saved_model.pb'
HD5_FILE_NAME = 'test.h5'
LAYERS_MODEL_NAME = 'model.json'


class CliTest(unittest.TestCase):
  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(CliTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(CliTest, self).tearDown()

  def _create_layers_model(self):
    data = {'format': 'layers-model'}
    filename = os.path.join(self._tmp_dir, 'model.json')
    with open(filename, 'a') as model_file:
      json.dump(data, model_file)

  def _create_hd5_file(self):
    input_tensor = tf.keras.layers.Input((3,))
    dense1 = tf.keras.layers.Dense(
        4, use_bias=True, kernel_initializer='ones', bias_initializer='zeros',
        name='MyDense10')(input_tensor)
    output = tf.keras.layers.Dense(
        2, use_bias=False, kernel_initializer='ones', name='MyDense20')(dense1)
    model = tf.keras.models.Model(inputs=[input_tensor], outputs=[output])
    h5_path = os.path.join(self._tmp_dir, HD5_FILE_NAME)
    print(h5_path)
    model.save_weights(h5_path)

  def _create_keras_saved_model(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape([2, 3], input_shape=[6]))
    model.add(tf.keras.layers.LSTM(10))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    tf.keras.models.save_model(model, save_dir)

  def _create_saved_model(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = tf.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    save.save(root, save_dir, to_save)

  def testOfValues(self):
    answers = {'input_path': 'abc', 'input_format': '123'}
    self.assertEqual(True, wizard.value_in_list(answers, 'input_path', ['abc']))
    self.assertEqual(False, wizard.value_in_list(answers,
                                                 'input_path', ['abd']))
    self.assertEqual(False, wizard.value_in_list(answers,
                                                 'input_format2', ['abc']))

  def testInputPathMessage(self):
    answers = {'input_format': 'keras'}
    self.assertEqual("The original path seems to be wrong, "
                     "what is the path of input HDF5 file?",
                     wizard.input_path_message(answers))

    answers = {'input_format': 'tf_hub'}
    self.assertEqual("The original path seems to be wrong, "
                     "what is the TFHub module URL? \n"
                     "(i.e. https://tfhub.dev/google/imagenet/"
                     "mobilenet_v1_100_224/classification/1)",
                     wizard.input_path_message(answers))

    answers = {'input_format': 'tf_saved_model'}
    self.assertEqual("The original path seems to be wrong, "
                     "what is the directory that contains the model?",
                     wizard.input_path_message(answers))

  def testValidateInputPathForTFHub(self):
    self.assertNotEqual(True,
                        wizard.validate_input_path(self._tmp_dir, 'tf_hub'))
    self.assertEqual(True,
                     wizard.validate_input_path("https://tfhub.dev/mobilenet",
                                                'tf_hub'))

  def testValidateInputPathForSavedModel(self):
    self.assertNotEqual(True, wizard.validate_input_path(
        self._tmp_dir, 'tf_saved_model'))
    self._create_saved_model()
    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'tf_saved_model'))

    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR, SAVED_MODEL_NAME)
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'tf_saved_model'))

  def testValidateInputPathForKerasSavedModel(self):
    self.assertNotEqual(True, wizard.validate_input_path(
        self._tmp_dir, 'keras_saved_model'))
    self._create_keras_saved_model()
    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'keras_saved_model'))

  def testValidateInputPathForKerasModel(self):
    self.assertNotEqual(True,
                        wizard.validate_input_path(self._tmp_dir, 'keras'))
    self._create_hd5_file()
    save_dir = os.path.join(self._tmp_dir, HD5_FILE_NAME)
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'keras'))

  def testValidateInputPathForLayersModel(self):
    self.assertNotEqual(True,
                        wizard.validate_input_path(self._tmp_dir, 'keras'))
    self._create_layers_model()
    save_dir = os.path.join(self._tmp_dir)
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'tfjs_layers_model'))

    save_dir = os.path.join(self._tmp_dir, 'model.json')
    self.assertEqual(True, wizard.validate_input_path(
        save_dir, 'tfjs_layers_model'))

  def testOutputPathExist(self):
    self.assertEqual(True, wizard.output_path_exists(self._tmp_dir))
    output_dir = os.path.join(self._tmp_dir, 'test')
    self.assertNotEqual(True, wizard.output_path_exists(output_dir))

  def testAvailableTags(self):
    self._create_saved_model()
    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    self.assertEqual(['serve'], wizard.available_tags(
        {'input_path': save_dir,
         'input_format': 'tf_saved_model'}))

  def testAvailableSignatureNames(self):
    self._create_saved_model()
    save_dir = os.path.join(self._tmp_dir, SAVED_MODEL_DIR)
    self.assertEqual(sorted(['__saved_model_init_op', 'serving_default']),
                     sorted(
                         [x['value'] for x in wizard.available_signature_names(
                             {'input_path': save_dir,
                              'input_format': 'tf_saved_model',
                              'saved_model_tags': 'serve'})]))

  def testGenerateCommandForSavedModel(self):
    options = {'input_format': 'tf_saved_model',
               'input_path': 'tmp/saved_model',
               'saved_model_tags': 'test',
               'signature_name': 'test_default',
               'quantize_float16': 'conv/*/weights',
               'weight_shard_size_bytes': '4194304',
               'skip_op_check': False,
               'strip_debug_ops': True,
               'control_flow_v2': True,
               'metadata': 'metadata_key:metadata.json',
               'output_path': 'tmp/web_model'}

    self.assertEqual(['--control_flow_v2=True',
                      '--input_format=tf_saved_model',
                      '--metadata=metadata_key:metadata.json',
                      '--quantize_float16=conv/*/weights',
                      '--saved_model_tags=test',
                      '--signature_name=test_default',
                      '--strip_debug_ops=True',
                      '--weight_shard_size_bytes=4194304',
                      'tmp/saved_model', 'tmp/web_model'],
                     wizard.generate_arguments(options))

  def testGenerateCommandForKerasSavedModel(self):
    options = {'input_format': 'tf_keras_saved_model',
               'output_format': 'tfjs_layers_model',
               'input_path': 'tmp/saved_model',
               'saved_model_tags': 'test',
               'signature_name': 'test_default',
               'weight_shard_size_bytes': '100',
               'quantize_float16': 'conv/*/weights',
               'skip_op_check': True,
               'strip_debug_ops': False,
               'control_flow_v2': False,
               'output_path': 'tmp/web_model'}

    self.assertEqual(['--control_flow_v2=False',
                      '--input_format=tf_keras_saved_model',
                      '--output_format=tfjs_layers_model',
                      '--quantize_float16=conv/*/weights',
                      '--saved_model_tags=test',
                      '--signature_name=test_default', '--skip_op_check',
                      '--strip_debug_ops=False',
                      '--weight_shard_size_bytes=100',
                      'tmp/saved_model', 'tmp/web_model'],
                     wizard.generate_arguments(options))

  def testGenerateCommandForKerasModel(self):
    options = {'input_format': 'keras',
               'input_path': 'tmp/model.HD5',
               'weight_shard_size_bytes': '100',
               'quantize_uint16': 'conv/*/weights',
               'output_path': 'tmp/web_model'}

    self.assertEqual(['--input_format=keras',
                      '--quantize_uint16=conv/*/weights',
                      '--weight_shard_size_bytes=100',
                      'tmp/model.HD5', 'tmp/web_model'],
                     wizard.generate_arguments(options))

  def testGenerateCommandForLayerModel(self):
    options = {'input_format': 'tfjs_layers_model',
               'output_format': 'keras',
               'input_path': 'tmp/model.json',
               'quantize_uint8': 'conv/*/weights',
               'weight_shard_size_bytes': '100',
               'output_path': 'tmp/web_model'}

    self.assertEqual(['--input_format=tfjs_layers_model',
                      '--output_format=keras',
                      '--quantize_uint8=conv/*/weights',
                      '--weight_shard_size_bytes=100',
                      'tmp/model.json',
                      'tmp/web_model'],
                     wizard.generate_arguments(options))


if __name__ == '__main__':
  unittest.main()
