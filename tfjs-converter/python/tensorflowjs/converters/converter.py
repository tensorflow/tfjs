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
"""Artifact conversion to and from Python TensorFlow and tf.keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import tempfile

import h5py
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflowjs import quantization
from tensorflowjs import version
from tensorflowjs.converters import common
from tensorflowjs.converters import keras_h5_conversion as conversion
from tensorflowjs.converters import keras_tfjs_loader
from tensorflowjs.converters import tf_saved_model_conversion_v2


def dispatch_keras_h5_to_tfjs_layers_model_conversion(
    h5_path, output_dir=None, quantization_dtype_map=None,
    split_weights_by_layer=False,
    weight_shard_size_bytes=1024 * 1024 * 4):
  """Converts a Keras HDF5 saved-model file to TensorFlow.js format.

  Auto-detects saved_model versus weights-only and generates the correct
  json in either case. This function accepts Keras HDF5 files in two formats:
    - A weights-only HDF5 (e.g., generated with Keras Model's `save_weights()`
      method),
    - A topology+weights combined HDF5 (e.g., generated with
      `tf.keras.model.save_model`).

  Args:
    h5_path: path to an HDF5 file containing keras model data as a `str`.
    output_dir: Output directory to which the TensorFlow.js-format model JSON
      file and weights files will be written. If the directory does not exist,
      it will be created.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.
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

  h5_file = h5py.File(h5_path, 'r')
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
        model_json, groups, output_dir, quantization_dtype_map,
        weight_shard_size_bytes=weight_shard_size_bytes)

  return model_json, groups


def dispatch_keras_h5_to_tfjs_graph_model_conversion(
    h5_path, output_dir=None,
    quantization_dtype_map=None,
    skip_op_check=False,
    strip_debug_ops=False,
    weight_shard_size_bytes=1024 * 1024 * 4,
    control_flow_v2=False):
  """
  Convert a keras HDF5-format model to tfjs GraphModel artifacts.

  Args:
    h5_path: Path to the HDF5-format file that contains the model saved from
      keras or tf.keras.
    output_dir: The destination to which the tfjs GraphModel artifacts will be
      written.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
  """

  if not os.path.exists(h5_path):
    raise ValueError('Nonexistent path to HDF5 file: %s' % h5_path)
  if os.path.isdir(h5_path):
    raise ValueError(
        'Expected path to point to an HDF5 file, but it points to a '
        'directory: %s' % h5_path)

  temp_savedmodel_dir = tempfile.mktemp(suffix='.savedmodel')
  model = tf.keras.models.load_model(h5_path, compile=False)
  model.save(temp_savedmodel_dir, include_optimizer=False, save_format='tf')

  # NOTE(cais): This cannot use `tf.compat.v1` because
  #   `convert_tf_saved_model()` works only in v2.
  tf_saved_model_conversion_v2.convert_tf_saved_model(
      temp_savedmodel_dir, output_dir,
      signature_def='serving_default',
      saved_model_tags='serve',
      quantization_dtype_map=quantization_dtype_map,
      skip_op_check=skip_op_check,
      strip_debug_ops=strip_debug_ops,
      weight_shard_size_bytes=weight_shard_size_bytes,
      control_flow_v2=control_flow_v2)

  # Clean up the temporary SavedModel directory.
  shutil.rmtree(temp_savedmodel_dir)


def dispatch_keras_saved_model_to_tensorflowjs_conversion(
    keras_saved_model_path, output_dir, quantization_dtype_map=None,
    split_weights_by_layer=False,
    weight_shard_size_bytes=1024 * 1024 * 4):
  """Converts keras model saved in the SavedModel format to tfjs format.

  Note that the SavedModel format exists in keras, but not in
  keras-team/tf.keras.

  Args:
    keras_saved_model_path: path to a folder in which the
      assets/saved_model.json can be found. This is usually a subfolder
      that is under the folder passed to
      `tf.keras.models.save_model()` and has a Unix epoch time
      as its name (e.g., 1542212752).
    output_dir: Output directory to which the TensorFlow.js-format model JSON
      file and weights files will be written. If the directory does not exist,
      it will be created.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.
    split_weights_by_layer: Whether to split the weights into separate weight
      groups (corresponding to separate binary weight files) layer by layer
      (Default: `False`).
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
  """
  with tf.Graph().as_default(), tf.compat.v1.Session():
    model = tf.keras.models.load_model(keras_saved_model_path)

    # Save model temporarily in HDF5 format.
    temp_h5_path = tempfile.mktemp(suffix='.h5')
    model.save(temp_h5_path, save_format='h5')
    assert os.path.isfile(temp_h5_path)

    dispatch_keras_h5_to_tfjs_layers_model_conversion(
        temp_h5_path,
        output_dir,
        quantization_dtype_map=quantization_dtype_map,
        split_weights_by_layer=split_weights_by_layer,
        weight_shard_size_bytes=weight_shard_size_bytes)

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
    tf.keras.models.save_model(
        model, keras_saved_model_path, save_format='tf')


def dispatch_tensorflowjs_to_tensorflowjs_conversion(
    config_json_path,
    output_dir_path,
    quantization_dtype_map=None,
    weight_shard_size_bytes=1024 * 1024 * 4):
  """Converts a Keras Model from tensorflowjs format to H5.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    output_dir_path: Path to output directory in which the result of the
      conversion will be saved.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.
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
        quantization_dtype_map=quantization_dtype_map,
        weight_shard_size_bytes=weight_shard_size_bytes)
    # TODO(cais): Support weight quantization.

  # Clean up the temporary H5 file.
  os.remove(temp_h5_path)


def dispatch_tfjs_layers_model_to_tfjs_graph_conversion(
    config_json_path,
    output_dir_path,
    quantization_dtype_map=None,
    skip_op_check=False,
    strip_debug_ops=False,
    weight_shard_size_bytes=1024 * 1024 * 4):
  """Converts a TensorFlow.js Layers Model to TensorFlow.js Graph Model.

  This conversion often benefits speed of inference, due to the graph
  optimization that goes into generating the Graph Model.

  Args:
    config_json_path: Path to the JSON file that includes the model's
      topology and weights manifest, in tensorflowjs format.
    output_dir_path: Path to output directory in which the result of the
      conversion will be saved.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
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

  model = keras_tfjs_loader.load_keras_model(config_json_path)
  model.save(temp_h5_path)
  dispatch_keras_h5_to_tfjs_graph_model_conversion(
      temp_h5_path, output_dir_path,
      quantization_dtype_map=quantization_dtype_map,
      skip_op_check=skip_op_check,
      strip_debug_ops=strip_debug_ops,
      weight_shard_size_bytes=weight_shard_size_bytes)

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
  input_format_is_keras = (
      input_format in [common.KERAS_MODEL, common.KERAS_SAVED_MODEL])
  input_format_is_tf = (
      input_format in [common.TF_SAVED_MODEL,
                       common.TF_FROZEN_MODEL, common.TF_HUB_MODEL])
  if output_format is None:
    # If no explicit output_format is provided, infer it from input format.
    if input_format_is_keras:
      output_format = common.TFJS_LAYERS_MODEL
    elif input_format_is_tf:
      output_format = common.TFJS_GRAPH_MODEL
    elif input_format == common.TFJS_LAYERS_MODEL:
      output_format = common.KERAS_MODEL

  return (input_format, output_format)

def _parse_quantization_dtype_map(float16, uint8, uint16, quantization_bytes):
  quantization_dtype_map = {}

  if quantization_bytes:
    print(
        'Warning: --quantization_bytes will be deprecated in a future release\n'
        'Please consider using --quantize_uint8, --quantize_uint16, '
        '--quantize_float16.', file=sys.stderr)
    if float16 is not None or uint8 is not None or uint16 is not None:
      raise ValueError(
          '--quantization_bytes cannot be used with the new quantization flags')

    dtype = quantization.QUANTIZATION_BYTES_TO_DTYPES[quantization_bytes]
    quantization_dtype_map[dtype] = True

  if float16 is not None:
    quantization_dtype_map[quantization.QUANTIZATION_DTYPE_FLOAT16] = \
      float16.split(',') if isinstance(float16, str) else float16
  if uint8 is not None:
    quantization_dtype_map[quantization.QUANTIZATION_DTYPE_UINT8] = \
      uint8.split(',') if isinstance(uint8, str) else uint8
  if uint16 is not None:
    quantization_dtype_map[quantization.QUANTIZATION_DTYPE_UINT16] = \
      uint16.split(',') if isinstance(uint16, str) else uint16

  return quantization_dtype_map

def get_arg_parser():
  """
  Create the argument parser for the converter binary.
  """

  parser = argparse.ArgumentParser('TensorFlow.js model converters.')
  parser.add_argument(
      common.INPUT_PATH,
      nargs='?',
      type=str,
      help='Path to the input file or directory. For input format "keras", '
      'an HDF5 (.h5) file is expected. For input format "tensorflow", '
      'a SavedModel directory, frozen model file, '
      'or TF-Hub module is expected.')
  parser.add_argument(
      common.OUTPUT_PATH,
      nargs='?',
      type=str,
      help='Path for all output artifacts.')
  parser.add_argument(
      '--%s' % common.INPUT_FORMAT,
      type=str,
      required=False,
      default=common.TF_SAVED_MODEL,
      choices=set([common.KERAS_MODEL, common.KERAS_SAVED_MODEL,
                   common.TF_SAVED_MODEL, common.TF_HUB_MODEL,
                   common.TFJS_LAYERS_MODEL, common.TF_FROZEN_MODEL]),
      help='Input format. '
      'For "keras", the input path can be one of the two following formats:\n'
      '  - A topology+weights combined HDF5 (e.g., generated with'
      '    `tf.keras.model.save_model()` method).\n'
      '  - A weights-only HDF5 (e.g., generated with Keras Model\'s '
      '    `save_weights()` method). \n'
      'For "keras_saved_model", the input_path must point to a subfolder '
      'under the saved model folder that is passed as the argument '
      'to tf.contrib.save_model.save_keras_model(). '
      'The subfolder is generated automatically by tensorflow when '
      'saving keras model in the SavedModel format. It is usually named '
      'as a Unix epoch time (e.g., 1542212752).\n'
      'For "tf" formats, a SavedModel, frozen model, '
      ' or TF-Hub module is expected.')
  parser.add_argument(
      '--%s' % common.OUTPUT_FORMAT,
      type=str,
      required=False,
      choices=set([common.KERAS_MODEL, common.KERAS_SAVED_MODEL,
                   common.TFJS_LAYERS_MODEL, common.TFJS_GRAPH_MODEL]),
      help='Output format. Default: tfjs_graph_model.')
  parser.add_argument(
      '--%s' % common.SIGNATURE_NAME,
      type=str,
      default=None,
      help='Signature of the SavedModel Graph or TF-Hub module to load. '
      'Applicable only if input format is "tf_hub" or "tf_saved_model".')
  parser.add_argument(
      '--%s' % common.SAVED_MODEL_TAGS,
      type=str,
      default='serve',
      help='Tags of the MetaGraphDef to load, in comma separated string '
      'format. Defaults to "serve". Applicable only if input format is '
      '"tf_saved_model".')
  parser.add_argument(
      '--%s' % common.QUANTIZATION_TYPE_FLOAT16,
      type=str,
      default=None,
      const=True,
      nargs='?',
      help='Comma separated list of node names to apply float16 quantization. '
      'You can also use wildcard symbol (*) to apply quantization to multiple '
      'nodes (e.g., conv/*/weights). When the flag is provided without any '
      'nodes the default behavior will match all nodes.')
  parser.add_argument(
      '--%s' % common.QUANTIZATION_TYPE_UINT8,
      type=str,
      default=None,
      const=True,
      nargs='?',
      help='Comma separated list of node names to apply 1-byte affine '
      'quantization. You can also use wildcard symbol (*) to apply '
      'quantization to multiple nodes (e.g., conv/*/weights). When the flag is '
      'provided without any nodes the default behavior will match all nodes.')
  parser.add_argument(
      '--%s' % common.QUANTIZATION_TYPE_UINT16,
      type=str,
      default=None,
      const=True,
      nargs='?',
      help='Comma separated list of node names to apply 2-byte affine '
      'quantization. You can also use wildcard symbol (*) to apply '
      'quantization to multiple nodes (e.g., conv/*/weights). When the flag is '
      'provided without any nodes the default behavior will match all nodes.')
  parser.add_argument(
      '--%s' % common.QUANTIZATION_BYTES,
      type=int,
      choices=set(quantization.QUANTIZATION_BYTES_TO_DTYPES.keys()),
      help='(Deprecated) How many bytes to optionally quantize/compress the '
      'weights to. 1- and 2-byte quantizaton is supported. The default '
      '(unquantized) size is 4 bytes.')
  parser.add_argument(
      '--%s' % common.SPLIT_WEIGHTS_BY_LAYER,
      action='store_true',
      help='Applicable to keras input_format only: Whether the weights from '
      'different layers are to be stored in separate weight groups, '
      'corresponding to separate binary weight files. Default: False.')
  parser.add_argument(
      '--%s' % common.VERSION,
      '-v',
      dest='show_version',
      action='store_true',
      help='Show versions of tensorflowjs and its dependencies')
  parser.add_argument(
      '--%s' % common.SKIP_OP_CHECK,
      action='store_true',
      help='Skip op validation for TensorFlow model conversion.')
  parser.add_argument(
      '--%s' % common.STRIP_DEBUG_OPS,
      type=bool,
      default=True,
      help='Strip debug ops (Print, Assert, CheckNumerics) from graph.')
  parser.add_argument(
      '--%s' % common.WEIGHT_SHARD_SIZE_BYTES,
      type=int,
      default=None,
      help='Shard size (in bytes) of the weight files. Currently applicable '
      'only when output_format is tfjs_layers_model or tfjs_graph_model.')
  parser.add_argument(
      '--output_node_names',
      type=str,
      help='The names of the output nodes, separated by commas. E.g., '
      '"logits,activations". Applicable only if input format is '
      '"tf_frozen_model".')
  parser.add_argument(
      '--%s' % common.CONTROL_FLOW_V2,
      type=str,
      help='Enable control flow v2 ops, this would improve inference '
      'performance on models with branches or loops.')
  return parser

def convert(arguments):
  args = get_arg_parser().parse_args(arguments)
  if args.show_version:
    print('\ntensorflowjs %s\n' % version.version)
    print('Dependency versions:')
    print('  keras %s' % tf.keras.__version__)
    print('  tensorflow %s' % tf.__version__)
    return

  if not args.input_path:
    raise ValueError(
        'Missing input_path argument. For usage, use the --help flag.')
  if not args.output_path:
    raise ValueError(
        'Missing output_path argument. For usage, use the --help flag.')

  if args.input_path is None:
    raise ValueError(
        'Error: The input_path argument must be set. '
        'Run with --help flag for usage information.')

  input_format, output_format = _standardize_input_output_formats(
      args.input_format, args.output_format)

  weight_shard_size_bytes = 1024 * 1024 * 4
  if args.weight_shard_size_bytes is not None:
    if (output_format not in
        (common.TFJS_LAYERS_MODEL, common.TFJS_GRAPH_MODEL)):
      raise ValueError(
          'The --weight_shard_size_bytes flag is only supported when '
          'output_format is tfjs_layers_model or tfjs_graph_model.')

    if not (isinstance(args.weight_shard_size_bytes, int) and
            args.weight_shard_size_bytes > 0):
      raise ValueError(
          'Expected weight_shard_size_bytes to be a positive integer, '
          'but got %s' % args.weight_shard_size_bytes)
    weight_shard_size_bytes = args.weight_shard_size_bytes

  quantization_dtype_map = _parse_quantization_dtype_map(
      args.quantize_float16,
      args.quantize_uint8,
      args.quantize_uint16,
      args.quantization_bytes
  )

  if (not args.output_node_names and input_format == common.TF_FROZEN_MODEL):
    raise ValueError(
        'The --output_node_names flag is required for "tf_frozen_model"')

  if (args.signature_name and input_format not in
      (common.TF_SAVED_MODEL, common.TF_HUB_MODEL)):
    raise ValueError(
        'The --signature_name flag is applicable only to "tf_saved_model" and '
        '"tf_hub" input format, but the current input format is '
        '"%s".' % input_format)

  if (args.control_flow_v2 and output_format != common.TFJS_GRAPH_MODEL):
    raise ValueError(
        'The --control_flow_v2 flag is applicable only to "tfjs_graph_model" '
        'as output format, but the current  output format '
        'is "%s"' % input_format, output_format)

  # TODO(cais, piyu): More conversion logics can be added as additional
  #   branches below.
  if (input_format == common.KERAS_MODEL and
      output_format == common.TFJS_LAYERS_MODEL):
    dispatch_keras_h5_to_tfjs_layers_model_conversion(
        args.input_path, output_dir=args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        split_weights_by_layer=args.split_weights_by_layer,
        weight_shard_size_bytes=weight_shard_size_bytes)
  elif (input_format == common.KERAS_MODEL and
        output_format == common.TFJS_GRAPH_MODEL):
    dispatch_keras_h5_to_tfjs_graph_model_conversion(
        args.input_path, output_dir=args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        skip_op_check=args.skip_op_check,
        strip_debug_ops=args.strip_debug_ops,
        weight_shard_size_bytes=weight_shard_size_bytes,
        control_flow_v2=args.control_flow_v2)
  elif (input_format == common.KERAS_SAVED_MODEL and
        output_format == common.TFJS_LAYERS_MODEL):
    dispatch_keras_saved_model_to_tensorflowjs_conversion(
        args.input_path, args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        split_weights_by_layer=args.split_weights_by_layer,
        weight_shard_size_bytes=weight_shard_size_bytes)
  elif (input_format == common.TF_SAVED_MODEL and
        output_format == common.TFJS_GRAPH_MODEL):
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        args.input_path, args.output_path,
        signature_def=args.signature_name,
        saved_model_tags=args.saved_model_tags,
        quantization_dtype_map=quantization_dtype_map,
        skip_op_check=args.skip_op_check,
        strip_debug_ops=args.strip_debug_ops,
        weight_shard_size_bytes=weight_shard_size_bytes,
        control_flow_v2=args.control_flow_v2)
  elif (input_format == common.TF_HUB_MODEL and
        output_format == common.TFJS_GRAPH_MODEL):
    tf_saved_model_conversion_v2.convert_tf_hub_module(
        args.input_path, args.output_path,
        signature=args.signature_name,
        saved_model_tags=args.saved_model_tags,
        quantization_dtype_map=quantization_dtype_map,
        skip_op_check=args.skip_op_check,
        strip_debug_ops=args.strip_debug_ops,
        weight_shard_size_bytes=weight_shard_size_bytes,
        control_flow_v2=args.control_flow_v2)
  elif (input_format == common.TFJS_LAYERS_MODEL and
        output_format == common.KERAS_MODEL):
    dispatch_tensorflowjs_to_keras_h5_conversion(args.input_path,
                                                 args.output_path)
  elif (input_format == common.TFJS_LAYERS_MODEL and
        output_format == common.KERAS_SAVED_MODEL):
    dispatch_tensorflowjs_to_keras_saved_model_conversion(args.input_path,
                                                          args.output_path)
  elif (input_format == common.TFJS_LAYERS_MODEL and
        output_format == common.TFJS_LAYERS_MODEL):
    dispatch_tensorflowjs_to_tensorflowjs_conversion(
        args.input_path, args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        weight_shard_size_bytes=weight_shard_size_bytes)
  elif (input_format == common.TFJS_LAYERS_MODEL and
        output_format == common.TFJS_GRAPH_MODEL):
    dispatch_tfjs_layers_model_to_tfjs_graph_conversion(
        args.input_path, args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        skip_op_check=args.skip_op_check,
        strip_debug_ops=args.strip_debug_ops,
        weight_shard_size_bytes=weight_shard_size_bytes)
  elif (input_format == common.TF_FROZEN_MODEL and
        output_format == common.TFJS_GRAPH_MODEL):
    tf_saved_model_conversion_v2.convert_tf_frozen_model(
        args.input_path, args.output_node_names, args.output_path,
        quantization_dtype_map=quantization_dtype_map,
        skip_op_check=args.skip_op_check,
        strip_debug_ops=args.strip_debug_ops,
        weight_shard_size_bytes=weight_shard_size_bytes)
  else:
    raise ValueError(
        'Unsupported input_format - output_format pair: %s - %s' %
        (input_format, output_format))

def pip_main():
  """Entry point for pip-packaged binary.

  Note that pip-packaged binary calls the entry method without
  any arguments, which is why this method is needed in addition to the
  `main` method below.
  """
  main([' '.join(sys.argv[1:])])


def main(argv):
  convert(argv[0].split(' '))


if __name__ == '__main__':
  tf1.app.run(main=main, argv=[' '.join(sys.argv[1:])])
