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

import keras
import tensorflow as tf

from tensorflowjs import read_weights
from tensorflowjs.converters import keras_h5_conversion


def load_keras_model(config_json_path,
                     weights_path_prefix=None,
                     weights_data_buffers=None,
                     load_weights=True,
                     use_unique_name_scope=False):
  """Load a Keras Model from TensorFlow.js-format artifacts.

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
    The loaded instance of `keras.Model`.

  Raises:
    TypeError, if the format of the JSON content of `config_json_path` has an
      invalid format.
    KeyError, if required keys do not exist in the JSON content of
      `config_json_path`.
    ValueError, if both `weights_data_buffers` and `weights_path_prefix` are
      provided.
  """
  with open(config_json_path, 'rt') as f:
    model_and_weights_manifest = json.load(f)

  if not isinstance(model_and_weights_manifest, dict):
    raise TypeError(
        'The JSON content of %s is required to be a `dict`, but found %s' %
        (config_json_path, type(model_and_weights_manifest)))
  if 'modelTopology' not in model_and_weights_manifest:
    raise KeyError(
        'Field "modelTopology" is missing from the JSON content in %s' %
        config_json_path)

  model_json = model_and_weights_manifest['modelTopology']

  if 'model_config' in model_json:
    model_json = model_json['model_config']
  unique_name_scope = uuid.uuid4().hex if use_unique_name_scope else None
  with tf.name_scope(unique_name_scope):
    model = keras.models.model_from_json(json.dumps(model_json))

  if load_weights:
    if 'weightsManifest' not in model_and_weights_manifest:
      raise KeyError(
          'Field "weightsManifest" is missing from the JSON content in %s' %
          config_json_path)
    weights_manifest = model_and_weights_manifest['weightsManifest']

    if weights_data_buffers:
      if weights_path_prefix:
        raise ValueError(
            'The arguments weights_data_buffers and weights_path_prefix are '
            'mutually exclusive and should not be both specified.')
      weight_entries = read_weights.decode_weights(weights_manifest,
                                                   weights_data_buffers,
                                                   flatten=True)
    else:
      weight_names = [
          keras_h5_conversion.normalize_weight_name(
              w.name[len(unique_name_scope) + 1:]) if use_unique_name_scope
          else keras_h5_conversion.normalize_weight_name(w.name[:-2])
          for w in model.weights]

      if not weights_path_prefix:
        weights_path_prefix = os.path.dirname(
            os.path.realpath(config_json_path))
      if not os.path.isdir(weights_path_prefix):
        raise ValueError(
            'Weights path prefix is not an existing directory: %s' %
            weights_path_prefix)

      weight_entries = read_weights.read_weights(weights_manifest,
                                                 weights_path_prefix,
                                                 flatten=True)
    weights_dict = dict()
    for weight_entry in weight_entries:
      weights_dict[weight_entry['name']] = weight_entry['data']

    weights_list = []
    for weight_name in weight_names:
      weights_list.append(weights_dict[weight_name])
    model.set_weights(weights_list)

  return model
