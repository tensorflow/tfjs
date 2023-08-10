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
"""Library for converting from hdf5 to json + binary weights.

Used primarily to convert saved weights, or saved_models from their
hdf5 format to a JSON + binary weights format that the TS codebase can use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

import six
import h5py
import numpy as np

from tensorflowjs import write_weights  # pylint: disable=import-error
from tensorflowjs.converters import common


def normalize_weight_name(weight_name):
  """Remove suffix ":0" (if present) from weight name."""
  name = as_text(weight_name)
  if name.endswith(':0'):
    # Python TensorFlow weight names ends with the output slot, which is
    # not applicable to TensorFlow.js.
    name = name[:-2]
  return name


def as_text(bytes_or_text, encoding='utf-8'):
  if isinstance(bytes_or_text, six.text_type):
    return bytes_or_text
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text.decode(encoding)
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    bytes_or_text)


def _convert_h5_group(group):
  """Construct a weights group entry.

  Args:
    group: The HDF5 group data, possibly nested.

  Returns:
    An array of weight groups (see `write_weights` in TensorFlow.js).
  """
  group_out = []
  if 'weight_names' in group.attrs:
    # This is a leaf node in namespace (e.g., 'Dense' in 'foo/bar/Dense').
    names = group.attrs['weight_names'].tolist()

    if not names:
      return group_out

    names = [as_text(name) for name in names]
    weight_values = [
        np.array(group[weight_name]) for weight_name in names]
    group_out += [{
        'name': normalize_weight_name(weight_name),
        'data': weight_value
    } for (weight_name, weight_value) in zip(names, weight_values)]
  else:
    # This is *not* a leaf level in the namespace (e.g., 'foo' in
    # 'foo/bar/Dense').
    for key in group.keys():
      # Call this method recursively.
      group_out += _convert_h5_group(group[key])

  return group_out

def _convert_v3_group(group, actual_layer_name):
  """Construct a weights group entry.

  Args:
    group: The HDF5 group data, possibly nested.

  Returns:
    An array of weight groups (see `write_weights` in TensorFlow.js).
  """
  group_out = []
  list_of_folder = [as_text(name) for name in group]
  if 'vars' in list_of_folder:
    names = group['vars']
    if not names:
      return group_out
    name_list = [as_text(name) for name in names]
    weight_values = [np.array(names[weight_name]) for weight_name in name_list]
    name_list = [os.path.join(actual_layer_name, item) for item in name_list]
    group_out += [{
    'name': normalize_weight_name(weight_name),
    'data': weight_value
    } for (weight_name, weight_value) in zip(name_list, weight_values)]
  else:
    for key in list_of_folder:
      group_out += _convert_v3_group(group[key], actual_layer_name)
  return group_out


def _check_version(h5file):
  """Check version compatiility.

  Args:
    h5file: An h5file object.

  Raises:
    ValueError: if the KerasVersion of the HDF5 file is unsupported.
  """
  keras_version = as_text(h5file.attrs['keras_version'])
  if keras_version.split('.')[0] != '2':
    raise ValueError(
        'Expected Keras version 2; got Keras version %s' % keras_version)


def _initialize_output_dictionary(h5file):
  """Prepopulate required fields for all data foramts.

  Args:
    h5file: Valid h5file object.

  Returns:
    A dictionary with common fields sets, shared across formats.
  """
  out = dict()
  out['keras_version'] = as_text(h5file.attrs['keras_version'])
  out['backend'] = as_text(h5file.attrs['backend'])
  return out


def _ensure_h5file(h5file):
  if not isinstance(h5file, h5py.File):
    return h5py.File(h5file, "r")
  else:
    return h5file


def _ensure_json_dict(item):
  return item if isinstance(item, dict) else json.loads(as_text(item))

def _discard_v3_keys(json_dict, keys_to_delete):
  if isinstance(json_dict, dict):
    keys = list(json_dict.keys())
    for key in keys:
      if key in keys_to_delete:
        del json_dict[key]
      else:
        _discard_v3_keys(json_dict[key], keys_to_delete)
  elif isinstance(json_dict, list):
    for item in json_dict:
      _discard_v3_keys(item, keys_to_delete)


# https://github.com/tensorflow/tfjs/issues/1255, b/124791387
# In tensorflow version 1.13 and some alpha and nightly-preview versions,
# the following layers have different class names in their serialization.
# This issue should be fixed in later releases. But we include the logic
# to translate them anyway, for users who use those versions of tensorflow.
_CLASS_NAME_MAP = {
    'BatchNormalizationV1': 'BatchNormalization',
    'UnifiedGRU': 'GRU',
    'UnifiedLSTM': 'LSTM'
}


def translate_class_names(input_object):
  """Perform class name replacement.

  Beware that this method modifies the input object in-place.
  """
  if not isinstance(input_object, dict):
    return
  for key in input_object:
    value = input_object[key]
    if key == 'class_name' and value in _CLASS_NAME_MAP:
      input_object[key] = _CLASS_NAME_MAP[value]
    elif isinstance(value, dict):
      translate_class_names(value)
    elif isinstance(value, (tuple, list)):
      for item in value:
        translate_class_names(item)


def h5_merged_saved_model_to_tfjs_format(h5file, split_by_layer=False):
  """Load topology & weight values from HDF5 file and convert.

  The HDF5 file is one generated by Keras' save_model method or model.save()

  N.B.:
  1) This function works only on HDF5 values from Keras version 2.
  2) This function does not perform conversion for special weights including
      ConvLSTM2D and CuDNNLSTM.

  Args:
    h5file: An instance of h5py.File, or the path to an h5py file.
    split_by_layer: (Optional) whether the weights of different layers are
      to be stored in separate weight groups (Default: `False`).

  Returns:
    (model_json, groups)
      model_json: a JSON dictionary holding topology and system metadata.
      group: an array of group_weights as defined in tfjs write_weights.

  Raises:
    ValueError: If the Keras version of the HDF5 file is not supported.
  """
  h5file = _ensure_h5file(h5file)
  try:
    _check_version(h5file)
  except ValueError:
    print("""failed to lookup keras version from the file,
    this is likely a weight only file""")
  model_json = _initialize_output_dictionary(h5file)

  model_json['model_config'] = _ensure_json_dict(
      h5file.attrs['model_config'])
  translate_class_names(model_json['model_config'])
  if 'training_config' in h5file.attrs:
    model_json['training_config'] = _ensure_json_dict(
        h5file.attrs['training_config'])

  groups = [] if split_by_layer else [[]]

  model_weights = h5file['model_weights']
  layer_names = [as_text(n) for n in model_weights]
  for layer_name in layer_names:
    layer = model_weights[layer_name]
    group = _convert_h5_group(layer)
    if group:
      if split_by_layer:
        groups.append(group)
      else:
        groups[0] += group
  return model_json, groups

def h5_v3_merged_saved_model_to_tfjs_format(h5file, meta_file, config_file,split_by_layer=False):
  """Load topology & weight values from HDF5 file and convert.

  The HDF5 weights file is one generated by Keras's save_model method or model.save()

  N.B.:
  1) This function works only on HDF5 values from Keras version 3.
  2) This function does not perform conversion for special weights including
      ConvLSTM2D and CuDNNLSTM.

  Args:
    h5file: An instance of h5py.File, or the path to an h5py file.
    split_by_layer: (Optional) whether the weights of different layers are
      to be stored in separate weight groups (Default: `False`).

  Returns:
    (model_json, groups)
      model_json: a JSON dictionary holding topology and system metadata.
      group: an array of group_weights as defined in tfjs write_weights.

  Raises:
    ValueError: If the Keras version of the HDF5 file is not supported.
  """
  h5file = _ensure_h5file(h5file)
  model_json = dict()
  model_json['keras_version'] = meta_file['keras_version']

  keys_to_remove = ["module", "registered_name", "date_saved"]
  config = _ensure_json_dict(config_file)
  _discard_v3_keys(config, keys_to_remove)
  model_json['model_config'] = config
  translate_class_names(model_json['model_config'])
  if 'training_config' in h5file.attrs:
    model_json['training_config'] = _ensure_json_dict(
        h5file.attrs['training_config'])

  groups = [] if split_by_layer else [[]]

  _convert_v3_group_structure_to_weights(groups=groups, group=h5file, split_by_layer=split_by_layer)
  return model_json, groups

def _convert_v3_group_structure_to_weights(groups, group, split_by_layer, indent=""):
  for key in group.keys():
    if isinstance(group[key], h5py.Group):
      _convert_v3_group_structure_to_weights(groups, group[key], split_by_layer, indent + key + "/")
    elif isinstance(group[key], h5py.Dataset):
      group_of_weights = dict()
      for key in group.keys():
        group_of_weights[str(indent + key)] = group[key]
      group_out = _convert_group(group_of_weights)
      if split_by_layer:
        groups.append(group_out)
      else:
        groups[0] += group_out
      break


def _convert_group(group_dict):
  group_out = []
  for key in group_dict.keys():
    name = key
    weights_value = np.array(group_dict[key])
    group_out += [{'name': name, 'data' : weights_value}]

  return group_out


def h5_weights_to_tfjs_format(h5file, split_by_layer=False):
  """Load weight values from a Keras HDF5 file and to a binary format.

  The HDF5 file is one generated by Keras' Model.save_weights() method.

  N.B.:
  1) This function works only on HDF5 values from Keras version 2.
  2) This function does not perform conversion for special weights including
      ConvLSTM2D and CuDNNLSTM.

  Args:
    h5file: An instance of h5py.File, or the path to an h5py file.
    split_by_layer: (Optional) whether the weights of different layers are
      to be stored in separate weight groups (Default: `False`).

  Returns:
    An array of group_weights as defined in tfjs write_weights.

  Raises:
    ValueError: If the Keras version of the HDF5 file is not supported
  """
  h5file = _ensure_h5file(h5file)
  try:
    _check_version(h5file)
  except ValueError:
    print("""failed to lookup keras version from the file,
    this is likely a weight only file""")

  groups = [] if split_by_layer else [[]]

  # pylint: disable=not-an-iterable
  layer_names = [as_text(n) for n in h5file.attrs['layer_names']]
  # pylint: enable=not-an-iterable
  for layer_name in layer_names:
    layer = h5file[layer_name]
    group = _convert_h5_group(layer)
    if group:
      if split_by_layer:
        groups.append(group)
      else:
        groups[0] += group
  return groups


def _get_generated_by(topology):
  if topology is None:
    return None
  elif 'keras_version' in topology:
    return 'keras v%s' % topology['keras_version']
  else:
    return None


def write_artifacts(topology,
                    weights,
                    output_dir,
                    quantization_dtype_map=None,
                    weight_shard_size_bytes=1024 * 1024 * 4,
                    metadata=None):
  """Writes weights and topology to the output_dir.

  If `topology` is Falsy (e.g., `None`), only emit weights to output_dir.

  Args:
    topology: a JSON dictionary, representing the Keras config.
    weights: an array of weight groups (as defined in tfjs write_weights).
    output_dir: the directory to hold all the contents.
    quantization_dtype_map: (Optional) A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    metadata: User defined metadata map.
  """
  # TODO(cais, nielsene): This method should allow optional arguments of
  #   `write_weights.write_weights` (e.g., shard size) and forward them.
  # We write the topology after since write_weights makes no promises about
  # preserving directory contents.
  if not (isinstance(weight_shard_size_bytes, int) and
          weight_shard_size_bytes > 0):
    raise ValueError(
        'Expected weight_shard_size_bytes to be a positive integer, '
        'but got %s' % weight_shard_size_bytes)

  if os.path.isfile(output_dir):
    raise ValueError(
        'Path "%d" already exists as a file (not a directory).' % output_dir)

  model_json = {
      common.FORMAT_KEY: common.TFJS_LAYERS_MODEL_FORMAT,
      common.GENERATED_BY_KEY: _get_generated_by(topology),
      common.CONVERTED_BY_KEY: common.get_converted_by()
  }

  if metadata:
    model_json[common.USER_DEFINED_METADATA_KEY] = metadata

  model_json[common.ARTIFACT_MODEL_TOPOLOGY_KEY] = topology or None
  weights_manifest = write_weights.write_weights(
      weights, output_dir, write_manifest=False,
      quantization_dtype_map=quantization_dtype_map,
      shard_size_bytes=weight_shard_size_bytes)
  assert isinstance(weights_manifest, list)
  model_json[common.ARTIFACT_WEIGHTS_MANIFEST_KEY] = weights_manifest

  model_json_path = os.path.join(
      output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)
  with open(model_json_path, 'wt') as f:
    json.dump(model_json, f)


def save_keras_model(model, artifacts_dir, quantization_dtype_map=None,
                     weight_shard_size_bytes=1024 * 1024 * 4, metadata=None):
  r"""Save a Keras model and its weights in TensorFlow.js format.

  Args:
    model: An instance of `keras.Model`.
    artifacts_dir: The directory in which the artifacts will be saved.
      The artifacts to be saved include:
        - model.json: A JSON representing the model. It has the following
          fields:
          - 'modelTopology': A JSON object describing the topology of the model,
            along with additional information such as training. It is obtained
            through calling `model.save()`.
          - 'weightsManifest': A TensorFlow.js-format JSON manifest for the
            model's weights.
        - files containing weight values in groups, with the file name pattern
          group(\d+)-shard(\d+)of(\d+).
      If the directory does not exist, this function will attempt to create it.
    quantization_dtype_map: (Optional) A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    metadata: User defined metadata map.

  Raises:
    ValueError: If `artifacts_dir` already exists as a file (not a directory).
  """
  temp_h5_path = tempfile.mktemp() + '.h5'
  model.save(temp_h5_path)
  topology_json, weight_groups = (
      h5_merged_saved_model_to_tfjs_format(temp_h5_path))
  if os.path.isfile(artifacts_dir):
    raise ValueError('Path "%s" already exists as a file.' % artifacts_dir)
  if not os.path.isdir(artifacts_dir):
    os.makedirs(artifacts_dir)
  write_artifacts(
      topology_json, weight_groups, artifacts_dir,
      quantization_dtype_map=quantization_dtype_map,
      weight_shard_size_bytes=weight_shard_size_bytes,
      metadata=metadata)
  os.remove(temp_h5_path)
