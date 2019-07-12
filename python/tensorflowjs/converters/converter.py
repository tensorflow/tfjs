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
import shutil
import tempfile

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflowjs import quantization
from tensorflowjs import version
from tensorflowjs.converters import keras_h5_conversion as conversion
from tensorflowjs.converters import keras_tfjs_loader
from tensorflowjs.converters import tf_saved_model_conversion_v2


def dispatch_keras_h5_to_tfjs_layers_model_conversion(
    h5_path, output_dir=None, quantization_dtype=None,
    split_weights_by_layer=False,
    weight_shard_size_bytes=1024 * 1024 * 4):
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
    quantization_dtype: The quantized data type to store the weights in
      (Default: `None`).
    split_weights_by_layer: Whether to split the weights into separate weight
      groups (corresponding to separate binary weight files) layer by layer
      (Default: `False`).
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.

  Returns:
    (model_json, groups)
      model_json: a json dictionary (empty if unused) for model topology.
        If `h5_path` points to a weights-only HDF5 file, this return value
        will be `None`.
      groups: an array of weight_groups as defined in tfjs weights_writer.
  """
  if not os.path.exists(h5_path):
    raise ValueError('Nonexistent path to HDF5 file: %s' % h5_path)
  if os.path.isdir(h5_path):
    raise ValueError(
        'Expected path to point to an HDF5 file, but it points to a '
        'directory: %s' % h5_path)

  h5_file = h5py.File(h5_path)
  if 'layer_names' in h5_file.attrs:
    model_json = None
    groups = conversion.h5_weights_to_tfjs_format(
        h5_file, split_by_layer=split_weights_by_layer)
  else:
    model_json, groups = conversion.h5_merged_saved_model_to_tfjs_format(
        h5_file, split_by_layer=split_weights_by_layer)

  if output_dir:
    if os.path.isfile(output_dir):
      raise ValueError(
          'Output path "%s" already exists as a file' % output_dir)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    conversion.write_artifacts(
        model_json, groups, output_dir, quantization_dtype,
        weight_shard_size_bytes=weight_shard_size_bytes)

  return model_json, groups


def dispatch_keras_h5_to_tfjs_graph_model_conversion(
    h5_path, output_dir=None,
    quantization_dtype=None,
    skip_op_check=False,
    strip_debug_ops=False):
  """
  Convert a keras HDF5-format model to tfjs GraphModel artifacts.

  Args:
    h5_path: Path to the HDF5-format file that contains the model saved from
      keras or tf.keras.
    output_dir: The destination to which the tfjs GraphModel artifacts will be
      written.
    quantization_dtype: The quantized data type to store the weights in
      (Default: `None`).
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
  """

  if not os.path.exists(h5_path):
    raise ValueError('Nonexistent path to HDF5 file: %s' % h5_path)
  if os.path.isdir(h5_path):
    raise ValueError(
        'Expected path to point to an HDF5 file, but it points to a '
        'directory: %s' % h5_path)

  temp_savedmodel_dir = tempfile.mktemp(suffix='.savedmodel')
  model = keras.models.load_model(h5_path)
  keras.experimental.export_saved_model(
      model, temp_savedmodel_dir, serving_only=True)

  # NOTE(cais): This cannot use `tf.compat.v1` because
  #   `convert_tf_saved_model()` works only in v2.
  tf_saved_model_conversion_v2.convert_tf_saved_model(
      temp_savedmodel_dir, output_dir,
      signature_def='serving_default',
      saved_model_tags='serve',
      quantization_dtype=quantization_dtype,
      skip_op_check=skip_op_check,
      strip_debug_ops=strip_debug_ops)

  # Clean up the temporary SavedModel directory.
  shutil.rmtree(temp_savedmodel_dir)


def dispatch_keras_saved_model_to_tensorflowjs_conversion(
    keras_saved_model_path, output_dir, quantization_dtype=None,
    split_weights_by_layer=False):
  """Converts keras model saved in the SavedModel format to tfjs format.

  Note that the SavedModel format exists in keras, but not in
  keras-team/keras.

  Args:
    keras_saved_model_path: path to a folder in which the
      assets/saved_model.json can be found. This is usually a subfolder
      that is under the folder passed to
      `keras.experimental.export_saved_model()` and has a Unix epoch time
      as its name (e.g., 1542212752).
    output_dir: Output directory to which the TensorFlow.js-format model JSON
      file and weights files will be written. If the directory does not exist,
      it will be created.
    quantization_dtype: The quantized data type to store the weights in
      (Default: `None`).
    split_weights_by_layer: Whether to split the weights into separate weight
      groups (corresponding to separate binary weight files) layer by layer
      (Default: `False`).
  """
  with tf.Graph().as_default(), tf.compat.v1.Session():
    model = keras.experimental.load_from_saved_model(keras_saved_model_path)

    # Save model temporarily in HDF5 format.
    temp_h5_path = tempfile.mktemp(suffix='.h5')
    model.save(temp_h5_path)
    assert os.path.isfile(temp_h5_path)

    dispatch_keras_h5_to_tfjs_layers_model_conversion(
        temp_h5_path,
        output_dir,
        quantization_dtype=quantization_dtype,
        split_weights_by_layer=split_weights_by_layer)

    # Delete temporary .h5 file.
    os.remove(temp_h5_path)


def dispatch_tensorflowjs_to_keras_h5_conversion(config_json_path, h5_path):
  """Converts a TensorFlow.js Layers model format to Keras H5.

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
        'For input_type=tfjs_layers_model & output_format=keras, '
        'the input path should be a model.json '
        'file, but received a directory.')
  if os.path.isdir(h5_path):
    raise ValueError(
        'For input_type=tfjs_layers_model & output_format=keras, '
        'the output path should be the path to an HDF5 file, '
        'but received an existing directory (%s).' % h5_path)

  # Verify that config_json_path points to a JSON file.
  with open(config_json_path, 'rt') as f:
    try:
      json.load(f)
    except (ValueError, IOError):
      raise ValueError(
          'For input_type=tfjs_layers_model & output_format=keras, '
          'the input path is expected to contain valid JSON content, '
          'but cannot read valid JSON content from %s.' % config_json_path)

  with tf.Graph().as_default(), tf.compat.v1.Session():
    model = keras_tfjs_loader.load_keras_model(config_json_path)
    model.save(h5_path)


def dispatch_tensorflowjs_to_keras_saved_model_conversion(
    config_json_path, keras_saved_model_path):
  """Converts a TensorFlow.js Layers model format to a tf.keras SavedModel.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    keras_saved_model_path: Path for the to-be-created Keras SavedModel.

  Raises:
    ValueError, if `config_json_path` is not a path to a valid JSON
      file, or if h5_path points to an existing directory.
  """
  if os.path.isdir(config_json_path):
    raise ValueError(
        'For input_type=tfjs_layers_model & output_format=keras_saved_model, '
        'the input path should be a model.json '
        'file, but received a directory.')

  # Verify that config_json_path points to a JSON file.
  with open(config_json_path, 'rt') as f:
    try:
      json.load(f)
    except (ValueError, IOError):
      raise ValueError(
          'For input_type=tfjs_layers_model & output_format=keras, '
          'the input path is expected to contain valid JSON content, '
          'but cannot read valid JSON content from %s.' % config_json_path)

  with tf.Graph().as_default(), tf.compat.v1.Session():
    model = keras_tfjs_loader.load_keras_model(config_json_path)
    keras.experimental.export_saved_model(
        model, keras_saved_model_path, serving_only=True)


def dispatch_tensorflowjs_to_tensorflowjs_conversion(
    config_json_path,
    output_dir_path,
    quantization_dtype=None,
    weight_shard_size_bytes=1024 * 1024 * 4):
  """Converts a Keras Model from tensorflowjs format to H5.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    output_dir_path: Path to output directory in which the result of the
      conversion will be saved.
    quantization_dtype: The quantized data type to store the weights in
      (Default: `None`).
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.

  Raises:
    ValueError, if `config_json_path` is not a path to a valid JSON
      file, or if h5_path points to an existing directory.
    ValueError, if `output_dir_path` exists and is a file (instead of
      a directory).
  """
  if os.path.isdir(config_json_path):
    raise ValueError(
        'For input_type=tfjs_layers_model, '
        'the input path should be a model.json '
        'file, but received a directory.')
  # TODO(cais): Assert output_dir_path doesn't exist or is the path to
  # a directory (not a file).

  # Verify that config_json_path points to a JSON file.
  with open(config_json_path, 'rt') as f:
    try:
      json.load(f)
    except (ValueError, IOError):
      raise ValueError(
          'For input_type=tfjs_layers_model, '
          'the input path is expected to contain valid JSON content, '
          'but cannot read valid JSON content from %s.' % config_json_path)

  temp_h5_path = tempfile.mktemp(suffix='.h5')
  with tf.Graph().as_default(), tf.compat.v1.Session():
    model = keras_tfjs_loader.load_keras_model(config_json_path)
    model.save(temp_h5_path)
    dispatch_tensorflowjs_to_keras_h5_conversion(config_json_path, temp_h5_path)

  with tf.Graph().as_default(), tf.compat.v1.Session():
    dispatch_keras_h5_to_tfjs_layers_model_conversion(
        temp_h5_path, output_dir_path,
        quantization_dtype=quantization_dtype,
        weight_shard_size_bytes=weight_shard_size_bytes)
    # TODO(cais): Support weight quantization.

  # Clean up the temporary H5 file.
  os.remove(temp_h5_path)


def dispatch_tfjs_layers_model_to_tfjs_graph_conversion(
    config_json_path,
    output_dir_path,
    quantization_dtype=None,
    skip_op_check=False,
    strip_debug_ops=False):
  """Converts a TensorFlow.js Layers Model to TensorFlow.js Graph Model.

  This conversion often benefits speed of inference, due to the graph
  optimization that goes into generating the Graph Model.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    output_dir_path: Path to output directory in which the result of the
      conversion will be saved.
    quantization_dtype: The quantized data type to store the weights in
      (Default: `None`).
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.

  Raises:
    ValueError, if `config_json_path` is not a path to a valid JSON
      file, or if h5_path points to an existing directory.
    ValueError, if `output_dir_path` exists and is a file (instead of
      a directory).
  """
  if os.path.isdir(config_json_path):
    raise ValueError(
        'For input_type=tfjs_layers_model, '
        'the input path should be a model.json '
        'file, but received a directory.')
  # TODO(cais): Assert output_dir_path doesn't exist or is the path to
  # a directory (not a file).

  # Verify that config_json_path points to a JSON file.
  with open(config_json_path, 'rt') as f:
    try:
      json.load(f)
    except (ValueError, IOError):
      raise ValueError(
          'For input_type=tfjs_layers_model, '
          'the input path is expected to contain valid JSON content, '
          'but cannot read valid JSON content from %s.' % config_json_path)

  temp_h5_path = tempfile.mktemp(suffix='.h5')

  model = keras_tfjs_loader.load_keras_model(config_json_path)
  model.save(temp_h5_path)
  dispatch_keras_h5_to_tfjs_graph_model_conversion(
      temp_h5_path, output_dir_path,
      quantization_dtype=quantization_dtype,
      skip_op_check=skip_op_check,
      strip_debug_ops=strip_debug_ops)

  # Clean up temporary HDF5 file.
  os.remove(temp_h5_path)


def _standardize_input_output_formats(input_format, output_format):
  """Standardize input and output formats.

  Args:
    input_format: Input format as a string.
    output_format: Output format as a string.

  Returns:
    A `tuple` of two strings:
      (standardized_input_format, standardized_output_format).
  """
  # https://github.com/tensorflow/tfjs/issues/1292: Remove the logic for the
  # explicit error message of the deprecated model type name 'tensorflowjs'
  # at version 1.1.0.
  if input_format == 'tensorflowjs':
    raise ValueError(
        '--input_format=tensorflowjs has been deprecated. '
        'Use --input_format=tfjs_layers_model instead.')

  input_format_is_keras = (
      input_format in ['keras', 'keras_saved_model'])
  input_format_is_tf = (
      input_format in ['tf_saved_model', 'tf_hub'])
  if output_format is None:
    # If no explicit output_format is provided, infer it from input format.
    if input_format_is_keras:
      output_format = 'tfjs_layers_model'
    elif input_format_is_tf:
      output_format = 'tfjs_graph_model'
    elif input_format == 'tfjs_layers_model':
      output_format = 'keras'
  elif output_format == 'tensorflowjs':
    # https://github.com/tensorflow/tfjs/issues/1292: Remove the logic for the
    # explicit error message of the deprecated model type name 'tensorflowjs'
    # at version 1.1.0.
    if input_format_is_keras:
      raise ValueError(
          '--output_format=tensorflowjs has been deprecated under '
          '--input_format=%s. Use --output_format=tfjs_layers_model '
          'instead.' % input_format)
    if input_format_is_tf:
      raise ValueError(
          '--output_format=tensorflowjs has been deprecated under '
          '--input_format=%s. Use --output_format=tfjs_graph_model '
          'instead.' % input_format)

  return (input_format, output_format)


def _parse_quantization_bytes(quantization_bytes):
  if quantization_bytes is None:
    return None
  elif quantization_bytes == 1:
    return np.uint8
  elif quantization_bytes == 2:
    return np.uint16
  else:
    raise ValueError('Unsupported quantization bytes: %s' % quantization_bytes)


def setup_arguments():
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
      choices=set(['keras', 'keras_saved_model',
                   'tf_saved_model', 'tf_hub', 'tfjs_layers_model',
                   'tensorflowjs']),
      help='Input format. '
      'For "keras", the input path can be one of the two following formats:\n'
      '  - A topology+weights combined HDF5 (e.g., generated with'
      '    `keras.model.save_model()` method).\n'
      '  - A weights-only HDF5 (e.g., generated with Keras Model\'s '
      '    `save_weights()` method). \n'
      'For "keras_saved_model", the input_path must point to a subfolder '
      'under the saved model folder that is passed as the argument '
      'to tf.contrib.save_model.save_keras_model(). '
      'The subfolder is generated automatically by tensorflow when '
      'saving keras model in the SavedModel format. It is usually named '
      'as a Unix epoch time (e.g., 1542212752).\n'
      'For "tf" formats, a SavedModel, frozen model, session bundle model, '
      ' or TF-Hub module is expected.')
  parser.add_argument(
      '--output_format',
      type=str,
      required=False,
      choices=set(['keras', 'keras_saved_model', 'tfjs_layers_model',
                   'tfjs_graph_model', 'tensorflowjs']),
      help='Output format. Default: tfjs_graph_model.')
  parser.add_argument(
      '--signature_name',
      type=str,
      default=None,
      help='Signature of the SavedModel Graph or TF-Hub module to load. '
      'Applicable only if input format is "tf_hub" or "tf_saved_model".')
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
      '--split_weights_by_layer',
      action='store_true',
      help='Applicable to keras input_format only: Whether the weights from '
      'different layers are to be stored in separate weight groups, '
      'corresponding to separate binary weight files. Default: False.')
  parser.add_argument(
      '--version',
      '-v',
      dest='show_version',
      action='store_true',
      help='Show versions of tensorflowjs and its dependencies')
  parser.add_argument(
      '--skip_op_check',
      action='store_true',
      help='Skip op validation for TensorFlow model conversion.')
  parser.add_argument(
      '--strip_debug_ops',
      type=bool,
      default=True,
      help='Strip debug ops (Print, Assert, CheckNumerics) from graph.')
  parser.add_argument(
      '--weight_shard_size_bytes',
      type=int,
      default=None,
      help='Shard size (in bytes) of the weight files. Currently applicable '
      'only to output_format=tfjs_layers_model.')
  return parser.parse_args()


def main():
  FLAGS = setup_arguments()
  if FLAGS.show_version:
    print('\ntensorflowjs %s\n' % version.version)
    print('Dependency versions:')
    print('  keras %s' % keras.__version__)
    print('  tensorflow %s' % tf.__version__)
    return

  weight_shard_size_bytes = 1024 * 1024 * 4
  if FLAGS.weight_shard_size_bytes:
    if  FLAGS.output_format != 'tfjs_layers_model':
      raise ValueError(
          'The --weight_shard_size_byte flag is only supported under '
          'output_format=tfjs_layers_model.')
    weight_shard_size_bytes = FLAGS.weight_shard_size_bytes

  if FLAGS.input_path is None:
    raise ValueError(
        'Error: The input_path argument must be set. '
        'Run with --help flag for usage information.')

  input_format, output_format = _standardize_input_output_formats(
      FLAGS.input_format, FLAGS.output_format)

  quantization_dtype = (
      quantization.QUANTIZATION_BYTES_TO_DTYPES[FLAGS.quantization_bytes]
      if FLAGS.quantization_bytes else None)

  if (FLAGS.signature_name and input_format not in
      ('tf_saved_model', 'tf_hub')):
    raise ValueError(
        'The --signature_name flag is applicable only to "tf_saved_model" and '
        '"tf_hub" input format, but the current input format is '
        '"%s".' % input_format)

  # TODO(cais, piyu): More conversion logics can be added as additional
  #   branches below.
  if input_format == 'keras' and output_format == 'tfjs_layers_model':
    dispatch_keras_h5_to_tfjs_layers_model_conversion(
        FLAGS.input_path, output_dir=FLAGS.output_path,
        quantization_dtype=quantization_dtype,
        split_weights_by_layer=FLAGS.split_weights_by_layer)
  elif input_format == 'keras' and output_format == 'tfjs_graph_model':
    dispatch_keras_h5_to_tfjs_graph_model_conversion(
        FLAGS.input_path, output_dir=FLAGS.output_path,
        quantization_dtype=quantization_dtype,
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)
  elif (input_format == 'keras_saved_model' and
        output_format == 'tfjs_layers_model'):
    dispatch_keras_saved_model_to_tensorflowjs_conversion(
        FLAGS.input_path, FLAGS.output_path,
        quantization_dtype=quantization_dtype,
        split_weights_by_layer=FLAGS.split_weights_by_layer)
  elif (input_format == 'tf_saved_model' and
        output_format == 'tfjs_graph_model'):
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        FLAGS.input_path, FLAGS.output_path,
        signature_def=FLAGS.signature_name,
        saved_model_tags=FLAGS.saved_model_tags,
        quantization_dtype=quantization_dtype,
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)
  elif (input_format == 'tf_hub' and
        output_format == 'tfjs_graph_model'):
    tf_saved_model_conversion_v2.convert_tf_hub_module(
        FLAGS.input_path, FLAGS.output_path, FLAGS.signature_name,
        FLAGS.saved_model_tags, skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)
  elif (input_format == 'tfjs_layers_model' and
        output_format == 'keras'):
    dispatch_tensorflowjs_to_keras_h5_conversion(FLAGS.input_path,
                                                 FLAGS.output_path)
  elif (input_format == 'tfjs_layers_model' and
        output_format == 'keras_saved_model'):
    dispatch_tensorflowjs_to_keras_saved_model_conversion(FLAGS.input_path,
                                                          FLAGS.output_path)
  elif (input_format == 'tfjs_layers_model' and
        output_format == 'tfjs_layers_model'):
    dispatch_tensorflowjs_to_tensorflowjs_conversion(
        FLAGS.input_path, FLAGS.output_path,
        quantization_dtype=_parse_quantization_bytes(FLAGS.quantization_bytes),
        weight_shard_size_bytes=weight_shard_size_bytes)
  elif (input_format == 'tfjs_layers_model' and
        output_format == 'tfjs_graph_model'):
    dispatch_tfjs_layers_model_to_tfjs_graph_conversion(
        FLAGS.input_path, FLAGS.output_path,
        quantization_dtype=_parse_quantization_bytes(FLAGS.quantization_bytes),
        skip_op_check=FLAGS.skip_op_check,
        strip_debug_ops=FLAGS.strip_debug_ops)
  else:
    raise ValueError(
        'Unsupported input_format - output_format pair: %s - %s' %
        (input_format, output_format))


if __name__ == '__main__':
  main()
