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
"""Artifact conversion to and from Python TensorFlow and Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import h5py
import keras
import tensorflow as tf

from tensorflowjs import quantization
from tensorflowjs import version
from tensorflowjs.converters import keras_h5_conversion
from tensorflowjs.converters import keras_tfjs_loader
from tensorflowjs.converters import tf_saved_model_conversion


def dispatch_keras_h5_to_tensorflowjs_conversion(
    h5_path, output_dir=None, quantization_dtype=None):
  """Converts a Keras HDF5 saved-model file to TensorFlow.js format.

  Auto-detects saved_model versus weights-only and generates the correct
  json in either case. This function accepts Keras HDF5 files in two formats:
    - A weights-only HDF5 (e.g., generated with Keras Model's `save_weights()`
      method),
    - A topology+weights combined HDF5 (e.g., generated with
      `keras.model.save_model`).

  Args:
    h5_path: path to an HDF5 file containing keras model data as a `str`.
    output_dir: Output directory to which the TensorFlow.js-format model JSON
      file and weights files will be written. If the directory does not exist,
      it will be created.

  Returns:
    (model_json, groups)
      model_json: a json dictionary (empty if unused) for model topology.
        If `h5_path` points to a weights-only HDF5 file, this return value
        will be `None`.
      groups: an array of weight_groups as defined in tfjs weights_writer.
  """
  converter = keras_h5_conversion.HDF5Converter()

  h5_file = h5py.File(h5_path)
  if 'layer_names' in h5_file.attrs:
    model_json = None
    groups = converter.h5_weights_to_tfjs_format(h5_file)
  else:
    model_json, groups = converter.h5_merged_saved_model_to_tfjs_format(
        h5_file)

  if output_dir:
    if os.path.isfile(output_dir):
      raise ValueError(
          'Output path "%s" already exists as a file' % output_dir)
    elif not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    converter.write_artifacts(
        model_json, groups, output_dir, quantization_dtype)

  return model_json, groups


def dispatch_tensorflowjs_to_keras_h5_conversion(config_json_path, h5_path):
  """Converts a Keras Model from tensorflowjs format to H5.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    h5_path: Path for the to-be-created Keras HDF5 model file.

  Raises:
    ValueError, if `config_json_path` is not a path to a valid JSON
      file, or if h5_path points to an existing directory.
  """
  if os.path.isdir(config_json_path):
    raise ValueError(
        'For input_type=tensorflowjs & output_format=keras, '
        'the input path should be a model.json '
        'file, but received a directory.')
  if os.path.isdir(h5_path):
    raise ValueError(
        'For input_type=tensorflowjs & output_format=keras, '
        'the output path should be the path to an HDF5 file, '
        'but received an existing directory (%s).' % h5_path)

  # Verify that config_json_path points to a JSON file.
  with open(config_json_path, 'rt') as f:
    try:
      json.load(f)
    except (ValueError, IOError):
      raise ValueError(
          'For input_type=tensorflowjs & output_format=keras, '
          'the input path is expected to contain valid JSON content, '
          'but cannot read valid JSON content from %s.' % config_json_path)

  with tf.Graph().as_default(), tf.Session():
    model = keras_tfjs_loader.load_keras_model(config_json_path)
    model.save(h5_path)
    print('Saved Keras model to HDF5 file: %s' % h5_path)


def main():
  parser = argparse.ArgumentParser('TensorFlow.js model converters.')
  parser.add_argument(
      'input_path',
      nargs='?',
      type=str,
      help='Path to the input file or directory. For input format "keras", '
      'an HDF5 (.h5) file is expected. For input format "tensorflow", '
      'a SavedModel directory, session bundle directory, frozen model file, '
      'or TF-Hub module is expected.')
  parser.add_argument(
      'output_path', nargs='?', type=str, help='Path for all output artifacts.')
  parser.add_argument(
      '--input_format',
      type=str,
      required=False,
      default='tf_saved_model',
      choices=set(['keras', 'tf_saved_model', 'tf_session_bundle',
                   'tf_frozen_model', 'tf_hub', 'tensorflowjs']),
      help='Input format. '
      'For "keras", the input path can be one of the two following formats:\n'
      '  - A topology+weights combined HDF5 (e.g., generated with'
      '    `keras.model.save_model()` method).\n'
      '  - A weights-only HDF5 (e.g., generated with Keras Model\'s '
      '    `save_weights()` method). \n'
      'For "tf" formats, a SavedModel, frozen model, session bundle model, '
      ' or TF-Hub module is expected.')
  parser.add_argument(
      '--output_format',
      type=str,
      required=False,
      choices=set(['keras', 'tensorflowjs']),
      default='tensorflowjs',
      help='Output format. Default: tensorflowjs.')
  parser.add_argument(
      '--output_node_names',
      type=str,
      help='The names of the output nodes, separated by commas. E.g., '
      '"logits,activations". Applicable only if input format is '
      '"tf_saved_model" or "tf_session_bundle".')
  parser.add_argument(
      '--signature_name',
      type=str,
      help='Signature of the TF-Hub module to load. Applicable only if input'
      ' format is "tf_hub".')
  parser.add_argument(
      '--saved_model_tags',
      type=str,
      default='serve',
      help='Tags of the MetaGraphDef to load, in comma separated string '
      'format. Defaults to "serve". Applicable only if input format is '
      '"tf_saved_model".')
  parser.add_argument(
      '--quantization_bytes',
      type=int,
      choices=set(quantization.QUANTIZATION_BYTES_TO_DTYPES.keys()),
      help='How many bytes to optionally quantize/compress the weights to. 1- '
      'and 2-byte quantizaton is supported. The default (unquantized) size is '
      '4 bytes.')
  parser.add_argument(
      '--version',
      '-v',
      dest='show_version',
      action='store_true',
      help='Show versions of tensorflowjs and its dependencies')
  parser.add_argument(
      '--skip_op_check',
      type=bool,
      default=False,
      help='Skip op validation for TensorFlow model conversion.')
  parser.add_argument(
      '--strip_debug_ops',
      type=bool,
      default=True,
      help='Strip debug ops (Print, Assert, CheckNumerics) from graph.')

  FLAGS = parser.parse_args()

  if FLAGS.show_version:
    print('\ntensorflowjs %s\n' % version.version)
    print('Dependency versions:')
    print('  keras %s' % keras.__version__)
    print('  tensorflow %s' % tf.__version__)
    return

  if FLAGS.input_path is None:
    raise ValueError(
        'Error: The input_path argument must be set. '
        'Run with --help flag for usage information.')

  quantization_dtype = (
      quantization.QUANTIZATION_BYTES_TO_DTYPES[FLAGS.quantization_bytes]
      if FLAGS.quantization_bytes else None)

  if (FLAGS.output_node_names and
      FLAGS.input_format not in
      ('tf_saved_model', 'tf_session_bundle', 'tf_frozen_model')):
    raise ValueError(
        'The --output_node_names flag is applicable only to input formats '
        '"tf_saved_model", "tf_session_bundle" and "tf_frozen_model", '
        'but the current input format is "%s".' % FLAGS.input_format)

  if FLAGS.signature_name and FLAGS.input_format != 'tf_hub':
    raise ValueError(
        'The --signature_name is applicable only to "tf_hub" input format, '
        'but the current input format is "%s".' % FLAGS.input_format)

  # TODO(cais, piyu): More conversion logics can be added as additional
  #   branches below.
  if FLAGS.input_format == 'keras' and FLAGS.output_format == 'tensorflowjs':
    dispatch_keras_h5_to_tensorflowjs_conversion(
        FLAGS.input_path, output_dir=FLAGS.output_path,
        quantization_dtype=quantization_dtype)

  elif (FLAGS.input_format == 'tf_saved_model' and
        FLAGS.output_format == 'tensorflowjs'):
    tf_saved_model_conversion.convert_tf_saved_model(
        FLAGS.input_path, FLAGS.output_node_names,
        FLAGS.output_path, saved_model_tags=FLAGS.saved_model_tags,
        quantization_dtype=quantization_dtype,
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)

  elif (FLAGS.input_format == 'tf_session_bundle' and
        FLAGS.output_format == 'tensorflowjs'):
    tf_saved_model_conversion.convert_tf_session_bundle(
        FLAGS.input_path, FLAGS.output_node_names,
        FLAGS.output_path, quantization_dtype=quantization_dtype,
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)

  elif (FLAGS.input_format == 'tf_frozen_model' and
        FLAGS.output_format == 'tensorflowjs'):
    tf_saved_model_conversion.convert_tf_frozen_model(
        FLAGS.input_path, FLAGS.output_node_names,
        FLAGS.output_path, quantization_dtype=quantization_dtype,
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)

  elif (FLAGS.input_format == 'tf_hub' and
        FLAGS.output_format == 'tensorflowjs'):
    if FLAGS.signature_name:
      tf_saved_model_conversion.convert_tf_hub_module(
          FLAGS.input_path, FLAGS.output_path, FLAGS.signature_name,
          skip_op_check=FLAGS.skip_op_check,
          strip_debug_ops=FLAGS.strip_debug_ops)
    else:
      tf_saved_model_conversion.convert_tf_hub_module(
          FLAGS.input_path,
          FLAGS.output_path,
          skip_op_check=FLAGS.skip_op_check,
          strip_debug_ops=FLAGS.strip_debug_ops)

  elif (FLAGS.input_format == 'tensorflowjs' and
        FLAGS.output_format == 'keras'):
    dispatch_tensorflowjs_to_keras_h5_conversion(FLAGS.input_path,
                                                 FLAGS.output_path)

  else:
    raise ValueError(
        'Unsupported input_format - output_format pair: %s - %s' %
        (FLAGS.input_format, FLAGS.output_format))


if __name__ == '__main__':
  main()
