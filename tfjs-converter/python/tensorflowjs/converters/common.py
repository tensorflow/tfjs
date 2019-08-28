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

import re

from tensorflow.python.framework import tensor_util

from tensorflowjs import version


# File name for the indexing JSON file in an artifact directory.
ARTIFACT_MODEL_JSON_FILE_NAME = 'model.json'

# JSON string keys for fields of the indexing JSON.
ARTIFACT_MODEL_TOPOLOGY_KEY = 'modelTopology'
ARTIFACT_WEIGHTS_MANIFEST_KEY = 'weightsManifest'

FORMAT_KEY = 'format'
TFJS_GRAPH_MODEL_FORMAT = 'graph-model'
TFJS_LAYERS_MODEL_FORMAT = 'layers-model'

GENERATED_BY_KEY = 'generatedBy'
CONVERTED_BY_KEY = 'convertedBy'

# Model formats.
KERAS_SAVED_MODEL = 'keras_saved_model'
KERAS_MODEL = 'keras'
TF_SAVED_MODEL = 'tf_saved_model'
TF_HUB_MODEL = 'tf_hub'
TFJS_GRAPH_MODEL = 'tfjs_graph_model'
TFJS_LAYERS_MODEL = 'tfjs_layers_model'

# CLI argument strings.
INPUT_PATH = 'input_path'
OUTPUT_PATH = 'output_path'
INPUT_FORMAT = 'input_format'
OUTPUT_FORMAT = 'output_format'
SIGNATURE_NAME = 'signature_name'
SAVED_MODEL_TAGS = 'saved_model_tags'
QUANTIZATION_BYTES = 'quantization_bytes'
SPLIT_WEIGHTS_BY_LAYER = 'split_weights_by_layer'
VERSION = 'version'
SKIP_OP_CHECK = 'skip_op_check'
STRIP_DEBUG_OPS = 'strip_debug_ops'
WEIGHT_SHARD_SIZE_BYTES = 'weight_shard_size_bytes'

def get_converted_by():
  """Get the convertedBy string for storage in model artifacts."""
  return 'TensorFlow.js Converter v%s' % version.version

def node_from_map(node_map, name):
  """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
  stripped_name = node_name_from_input(name)
  if stripped_name not in node_map:
    raise ValueError("No node named '%s' found in map." % name)
  return node_map[stripped_name]


def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Node named '%s' should be a Const op for values_from_const." %
        node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value

# Whether to scale by gamma after normalization.
def scale_after_normalization(node):
  if node.op == "BatchNormWithGlobalNormalization":
    return node.attr["scale_after_normalization"].b
  return True

def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name
