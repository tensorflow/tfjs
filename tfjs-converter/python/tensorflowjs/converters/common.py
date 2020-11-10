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
from tensorflowjs import version


# File name for the indexing JSON file in an artifact directory.
ARTIFACT_MODEL_JSON_FILE_NAME = 'model.json'

# JSON string keys for fields of the indexing JSON.
ARTIFACT_MODEL_TOPOLOGY_KEY = 'modelTopology'
ARTIFACT_MODEL_INITIALIZER = 'modelInitializer'
ARTIFACT_WEIGHTS_MANIFEST_KEY = 'weightsManifest'

FORMAT_KEY = 'format'
TFJS_GRAPH_MODEL_FORMAT = 'graph-model'
TFJS_LAYERS_MODEL_FORMAT = 'layers-model'

GENERATED_BY_KEY = 'generatedBy'
CONVERTED_BY_KEY = 'convertedBy'

SIGNATURE_KEY = 'signature'
USER_DEFINED_METADATA_KEY = 'userDefinedMetadata'

# Model formats.
KERAS_SAVED_MODEL = 'keras_saved_model'
KERAS_MODEL = 'keras'
TF_SAVED_MODEL = 'tf_saved_model'
TF_HUB_MODEL = 'tf_hub'
TFJS_GRAPH_MODEL = 'tfjs_graph_model'
TFJS_LAYERS_MODEL = 'tfjs_layers_model'
TF_FROZEN_MODEL = 'tf_frozen_model'

# CLI argument strings.
INPUT_PATH = 'input_path'
OUTPUT_PATH = 'output_path'
INPUT_FORMAT = 'input_format'
OUTPUT_FORMAT = 'output_format'
OUTPUT_NODE = 'output_node_names'
SIGNATURE_NAME = 'signature_name'
SAVED_MODEL_TAGS = 'saved_model_tags'
QUANTIZATION_BYTES = 'quantization_bytes'
QUANTIZATION_TYPE_FLOAT16 = 'quantize_float16'
QUANTIZATION_TYPE_UINT8 = 'quantize_uint8'
QUANTIZATION_TYPE_UINT16 = 'quantize_uint16'
SPLIT_WEIGHTS_BY_LAYER = 'split_weights_by_layer'
VERSION = 'version'
SKIP_OP_CHECK = 'skip_op_check'
STRIP_DEBUG_OPS = 'strip_debug_ops'
WEIGHT_SHARD_SIZE_BYTES = 'weight_shard_size_bytes'
CONTROL_FLOW_V2 = 'control_flow_v2'
EXPERIMENTS = 'experiments'
METADATA = 'metadata'

def get_converted_by():
  """Get the convertedBy string for storage in model artifacts."""
  return 'TensorFlow.js Converter v%s' % version.version
