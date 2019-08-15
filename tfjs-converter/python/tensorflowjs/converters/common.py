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
ARTIFACT_WEIGHTS_MANIFEST_KEY = 'weightsManifest'

FORMAT_KEY = 'format'
TFJS_GRAPH_MODEL_FORMAT = 'graph-model'
TFJS_LAYERS_MODEL_FORMAT = 'layers-model'

GENERATED_BY_KEY = 'generatedBy'
CONVERTED_BY_KEY = 'convertedBy'


def get_converted_by():
  """Get the convertedBy string for storage in model artifacts."""
  return 'TensorFlow.js Converter v%s' % version.version
