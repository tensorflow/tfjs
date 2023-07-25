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
"""Library for loading a Keras model from TensorFlow.js format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import uuid
import zipfile
import datetime
import six
import h5py
import keras_nlp
import tensorflow.compat.v2 as tf
from tensorflowjs.converters import tf_module_mapper
from tensorflowjs.converters import keras_h5_conversion
from tensorflowjs.converters.tf_module_mapper import TFCLASS_MODULE_MAP
from tensorflowjs import read_weights
from keras.src.saving import object_registration
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.saving_lib import ATTR_SKIPLIST
from keras.src.utils import generic_utils
from keras.src.engine import base_layer
from keras.src.optimizers import optimizer
from keras import losses

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"

def _deserialize_keras_model(model_topology_json,
                             weight_entries=None,
                             use_unique_name_scope=False):
  """Internal helper method for deserializing a Keras Model.

  Args:
    model_topology_json: content of the JSON containing model topology, in
      Keras (i.e., tfjs-layers) format. It can be any of the following types:
      - A JSON object, i.e., a `dict`.
      - A `str` or `buffer`, in which case it will be parsed as a JSON object.
      - A `file` object or `file`-like object containing the JSON, in which
        case it will be read with the `read()` method and the content parsed
        as a JSON object.
    weight_entries: Weight entries, in tensorflow.js format, as a `list`.
    use_unique_name_scope: Use a unique ID as the name scope for the loaded
      model. This may facilitate loading of multiple Keras models in the
      same TensorFlow Graph or Session context. Default: `False`.
  """
  if isinstance(model_topology_json, (six.string_types, bytes)):
    model_topology_json = json.loads(tf.compat.as_text(model_topology_json))
  elif not isinstance(model_topology_json, dict):
    model_topology_json = json.load(model_topology_json)

  if 'model_config' in model_topology_json:
    model_topology_json = model_topology_json['model_config']
  unique_name_scope = uuid.uuid4().hex if use_unique_name_scope else None
  with tf.compat.v1.name_scope(unique_name_scope):
    model = tf.keras.models.model_from_json(json.dumps(model_topology_json))

  if weight_entries:
    weights_dict = dict()
    for weight_entry in weight_entries:
      weights_dict[weight_entry['name']] = weight_entry['data']

    # Collect weight names from the model, in the same order as the internal
    # ordering of model.set_weights() used below.
    weight_names = []
    for layer in model.layers:
      for w in layer.weights:
        weight_names.append(
            keras_h5_conversion.normalize_weight_name(
                w.name[len(unique_name_scope) + 1:])
            if use_unique_name_scope
            else keras_h5_conversion.normalize_weight_name(w.name))

    # Prepare list of weight values for calling set_weights().
    weights_list = []

    for name in weight_names:
      if name in weights_dict:
        weights_list.append(weights_dict[name])
      else:
        # TF 2.2.0 added cell name to the weight name in the format of
        # layer_name/cell_name/weight_name, we need to remove
        # the inner cell name.
        tokens = name.split('/')
        shorten_name = '/'.join(tokens[0:-2] + [tokens[-1]])
        weights_list.append(weights_dict[shorten_name])

    model.set_weights(weights_list)

  return model

def _deserialize_keras_keras_model(model_topology_json,
                             weight_entries=None,
                             use_unique_name_scope=False):
  """Internal helper method for deserializing a Keras V3 Model.

  Args:
    model_topology_json: content of the JSON containing model topology, in
      Keras (i.e., tfjs-layers) format. It can be any of the following types:
      - A JSON object, i.e., a `dict`.
      - A `str` or `buffer`, in which case it will be parsed as a JSON object.
      - A `file` object or `file`-like object containing the JSON, in which
        case it will be read with the `read()` method and the content parsed
        as a JSON object.
    weight_entries: Weight entries, in tensorflow.js format, as a `list`.
    use_unique_name_scope: Use a unique ID as the name scope for the loaded
      model. This may facilitate loading of multiple Keras models in the
      same TensorFlow Graph or Session context. Default: `False`.
  """
  if isinstance(model_topology_json, (six.string_types, bytes)):
    model_topology_json = json.loads(tf.compat.as_text(model_topology_json))
  elif not isinstance(model_topology_json, dict):
    model_topology_json = json.load(model_topology_json)

  if 'model_config' in model_topology_json:
    # Build the map between class and its corresponding module in TF.
    if ('module' or 'registered_name') not in model_topology_json['model_config']:
      _generate_v3_keys(model_topology_json['model_config'])
    model_topology_json = model_topology_json['model_config']
  else:
    raise Exception("'model_config' does not exist in json file.")

  model = deserialize_keras_object(model_topology_json)
  if weight_entries:
    weights_dict = dict()
    for weight_entry in weight_entries:
      weights_dict[weight_entry['name']] = weight_entry['data']
  _load_state(model,
              weights_dict=weights_dict,
              inner_path="",
              visited_trackables=set())
  if not weights_dict:
    raise Exception('Unassigned weights for the model.')
  os.remove('temp.h5')
  return model

def _load_state(trackable, weights_dict, inner_path, visited_trackables=None):
  if visited_trackables and id(trackable) in visited_trackables:
    return

  if hasattr(trackable, 'load_own_variables') and weights_dict:
    filter_list = []
    for key, val in weights_dict.items():
      temp_str = key.split('/')
      for i in temp_str:
        # discard 'vars' in order to match with the path within the model.
        if i == 'vars':
          temp_str.remove(i)
      match_str = '/'.join(temp_str[:-1])
      if inner_path == match_str:
        filter_list.append(key)
    # Temporary .h5 file to store dataset.
    h5f = h5py.File('temp.h5', 'w')
    for i in filter_list:
      index = i.split('/')[-1]
      # Create temp dataset named with index.
      h5f.create_dataset(f'{index}', data=weights_dict[i])
      del weights_dict[i]
    trackable.load_own_variables(h5f)
    h5f.close()

  if visited_trackables is not None:
    visited_trackables.add(id(trackable))

  for child_attr, child_obj in _walk_trackable(trackable):
    if _is_keras_trackable(child_obj):
      _load_state(child_obj,
                  weights_dict,
                  inner_path=tf.io.gfile.join(inner_path, child_attr),
                  visited_trackables=visited_trackables)
    elif isinstance(child_obj, (list, dict, tuple, set)):
      _load_container_state(child_obj,
                            weights_dict,
                            inner_path=tf.io.gfile.join(inner_path, child_attr),
                            visited_trackables=visited_trackables)

def _walk_trackable(trackable):
  for child_attr in dir(trackable):
    if child_attr.startswith('__') or child_attr in ATTR_SKIPLIST:
      continue
    try:
      child_obj = getattr(trackable, child_attr)
    except Exception:
      continue
    yield child_attr, child_obj

def _is_keras_trackable(obj):
  from keras.src.metrics import base_metric

  return isinstance(
    obj,
    (base_layer.Layer,
     optimizer.Optimizer,
     base_metric.Metric,
     losses.Loss,
     ),
     )

def _load_container_state(
    container,
    weight_dict,
    inner_path,
    visited_trackables,
):
    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for trackable in container:
        if _is_keras_trackable(trackable):
            # Keeps layer name indexing in proper order
            # when duplicate layers are in container.
            if visited_trackables and id(trackable) in visited_trackables:
                continue
            # Do NOT address the trackable via `trackable.name`, since
            # names are usually autogenerated and thus not reproducible
            # (i.e. they may vary across two instances of the same model).
            name = generic_utils.to_snake_case(trackable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            _load_state(
                trackable=trackable,
                weights_dict=weight_dict,
                inner_path=tf.io.gfile.join(inner_path, name),
                visited_trackables=visited_trackables,
            )

def _check_config_json(config_json):
  if not isinstance(config_json, dict):
    raise TypeError(
        'The JSON content is required to be a `dict`, but found %s.' %
        type(config_json))
  if 'modelTopology' not in config_json:
    raise KeyError('Field "modelTopology" is missing from the JSON content.')


def _get_weights_manifest_from_config_json(config_json):
  if 'weightsManifest' not in config_json:
    raise KeyError(
        'Field "weightsManifest" is missing from the JSON content.')
  return config_json['weightsManifest']

def _generate_v3_keys(config):
  if isinstance(config, dict):
    list_of_keys = list(config.keys())
    for key in list_of_keys:
      _generate_v3_keys(config[key])
    if 'class_name' in list_of_keys:
      try:
        config['module'] = tf_module_mapper.get_module_path(config['class_name'])
        # Set registered name to None since it is not a custom class.
        config['registered_name'] = None
      except Exception:
        if config['class_name'] in keras_nlp.models.__dict__:
          obj = keras_nlp.models.__dict__.get(config['class_name'])
          config['module'] = obj.__module__
          # Set registered name of custom class in keras_nlp.
          config['registered_name'] = object_registration.get_registered_name(obj)
        else:
          raise KeyError(f"Unknown class name {config['class_name']}")

  elif isinstance(config, list):
    for item in config:
      _generate_v3_keys(item)


def deserialize_keras_model(config_json,
                            weight_data=None,
                            use_unique_name_scope=False):
  """Deserialize a Keras Model from buffers or file-like objects.

  Args:
    config: content of the JSON containing model topology and weights manifest,
      in Keras (i.e., tfjs-layers) format. It can be one of the following
      types:
      - A JSON object, i.e., a `dict`.
      - A `str` or `buffer`, in which case it will be parsed as a JSON object.
      - A `file` object or `file`-like object containing the JSON, in which
        case it will be read with the `read()` method and the content parsed
        as a JSON object.
    weight_data: a `list` of `buffer`s or a `list` of `file`-like objects
      (e.g., `io.BytesIO`) containing the binary weight values.
      If `None`, the weights of the model will not be loaded (i.e., only the
      topology of the model will be loaded).
    use_unique_name_scope: Use a unique ID as the name scope for the loaded
      model. This may facilitate loading of multiple Keras models in the
      same TensorFlow Graph or Session context. Default: `False`.
  """
  if isinstance(config_json, (six.string_types, bytes)):
    config_json = json.loads(tf.compat.as_text(config_json))
  elif not isinstance(config_json, dict):
    config_json = json.load(config_json)
  _check_config_json(config_json)
  model_topology_json = config_json['modelTopology']

  weight_entries = None
  if weight_data:
    weights_manifest = _get_weights_manifest_from_config_json(config_json)
    if not isinstance(weight_data, list):
      raise ValueError(
          'weight_data must be a list, but is %s' % type(weight_data))
    if hasattr(weight_data[0], 'read'):
      # weight_data is a list of file-like objects.
      weight_data = [d.read() for d in weight_data]
    weight_entries = read_weights.decode_weights(weights_manifest,
                                                 weight_data,
                                                 flatten=True)

  return _deserialize_keras_model(model_topology_json,
                                  weight_entries=weight_entries,
                                  use_unique_name_scope=use_unique_name_scope)


def load_keras_model(config_json_path,
                     weights_path_prefix=None,
                     weights_data_buffers=None,
                     load_weights=True,
                     use_unique_name_scope=False):
  """Load a Keras Model from TensorFlow.js-format artifacts from file system

  Args:
    config_json_path: Path to the TensorFlow.js-format JSON file that includes
      the model topology and weights manifest.
    weights_path_prefix: Optional path prefix for the weights files.
      If not specified (`None`), will assume the prefix is the same directory
      as the dirname of `config_json_path`.
    weights_data_buffers: A buffer of a `list` of buffers containing the weight
      values concatenated and sharded in the order as specified by the
      weights manifest at `config_json_path`. This argument is mutually
      exclusive with `weights_path_prefix`.
    load_weights: Whether the weights are to be loaded according
      to the weights manifest at `config_json_path`. Default: `True`.
    use_unique_name_scope: Use a unique ID as the name scope for the loaded
      model. This may facilitate loading of multiple Keras models in the
      same TensorFlow Graph or Session context. Default: `False`.

  Returns:
    The loaded instance of `tf.keras.Model`.

  Raises:
    TypeError, if the format of the JSON content of `config_json_path` has an
      invalid format.
    KeyError, if required keys do not exist in the JSON content of
      `config_json_path`.
    ValueError, if both `weights_data_buffers` and `weights_path_prefix` are
      provided.
  """
  if weights_data_buffers and weights_path_prefix:
    raise ValueError(
        'The arguments weights_data_buffers and weights_path_prefix are '
        'mutually exclusive and should not be both specified.')

  with open(config_json_path, 'rt') as f:
    config_json = json.load(f)
    _check_config_json(config_json)

  weight_entries = None
  if load_weights:
    weights_manifest = _get_weights_manifest_from_config_json(config_json)

    if not weights_data_buffers and not weights_path_prefix:
      weights_path_prefix = os.path.dirname(
          os.path.realpath(config_json_path))
    if not os.path.isdir(weights_path_prefix):
      raise ValueError(
          'Weights path prefix is not an existing directory: %s' %
          weights_path_prefix)
    if weights_path_prefix:
      weight_entries = read_weights.read_weights(weights_manifest,
                                                 weights_path_prefix,
                                                 flatten=True)
    else:
      weight_entries = read_weights.decode_weights(weights_manifest,
                                                   weights_data_buffers,
                                                   flatten=True)

  return _deserialize_keras_model(config_json['modelTopology'],
                                  weight_entries=weight_entries,
                                  use_unique_name_scope=use_unique_name_scope)

def load_keras_keras_model(config_json_path,
                     weights_path_prefix=None,
                     weights_data_buffers=None,
                     load_weights=True,
                     use_unique_name_scope=False):
  """Load a Keras Model from TensorFlow.js-format artifacts from file system

  Args:
    config_json_path: Path to the TensorFlow.js-format JSON file that includes
      the model topology and weights manifest.
    weights_path_prefix: Optional path prefix for the weights files.
      If not specified (`None`), will assume the prefix is the same directory
      as the dirname of `config_json_path`.
    weights_data_buffers: A buffer of a `list` of buffers containing the weight
      values concatenated and sharded in the order as specified by the
      weights manifest at `config_json_path`. This argument is mutually
      exclusive with `weights_path_prefix`.
    load_weights: Whether the weights are to be loaded according
      to the weights manifest at `config_json_path`. Default: `True`.
    use_unique_name_scope: Use a unique ID as the name scope for the loaded
      model. This may facilitate loading of multiple Keras models in the
      same TensorFlow Graph or Session context. Default: `False`.

  Returns:
    The loaded instance of `tf.keras.Model`.

  Raises:
    TypeError, if the format of the JSON content of `config_json_path` has an
      invalid format.
    KeyError, if required keys do not exist in the JSON content of
      `config_json_path`.
    ValueError, if both `weights_data_buffers` and `weights_path_prefix` are
      provided.
  """
  if weights_data_buffers and weights_path_prefix:
    raise ValueError(
        'The arguments weights_data_buffers and weights_path_prefix are '
        'mutually exclusive and should not be both specified.')

  with open(config_json_path, 'rt') as f:
    config_json = json.load(f)
    _check_config_json(config_json)

  weight_entries = None
  if load_weights:
    weights_manifest = _get_weights_manifest_from_config_json(config_json)

    if not weights_data_buffers and not weights_path_prefix:
      weights_path_prefix = os.path.dirname(
          os.path.realpath(config_json_path))
    if not os.path.isdir(weights_path_prefix):
      raise ValueError(
          'Weights path prefix is not an existing directory: %s' %
          weights_path_prefix)
    if weights_path_prefix:
      weight_entries = read_weights.read_weights(weights_manifest,
                                                 weights_path_prefix,
                                                 flatten=True)
    else:
      weight_entries = read_weights.decode_weights(weights_manifest,
                                                   weights_data_buffers,
                                                   flatten=True)

  return _deserialize_keras_keras_model(config_json['modelTopology'],
                                     weight_entries=weight_entries,
                                     use_unique_name_scope=use_unique_name_scope)
