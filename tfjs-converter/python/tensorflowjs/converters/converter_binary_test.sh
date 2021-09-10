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

set -e

GENERATE_BIN="${TEST_SRCDIR}/tfjs/tfjs-converter/python/tensorflowjs/converters/generate_test_model"
CONVERTER_BIN="${TEST_SRCDIR}/tfjs/tfjs-converter/python/tensorflowjs/converters/converter"

# 1. Test tf_saved_model --> tfjs_graph_model conversion.
SAVED_MODEL_DIR="$(mktemp -d)"
echo "Genearting TF SavedModel for testing..."
"${GENERATE_BIN}" "${SAVED_MODEL_DIR}" --model_type tf_saved_model
echo "Done genearting TF SavedModel for testing at ${SAVED_MODEL_DIR}"

OUTPUT_DIR="${SAVED_MODEL_DIR}_converted"
"${CONVERTER_BIN}" \
    --input_format tf_saved_model \
    --output_format tfjs_graph_model \
    "${SAVED_MODEL_DIR}" \
    "${OUTPUT_DIR}"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "ERROR: Failed to find conversion output directory: ${OUTPUT_DIR}" 1>&2
  exit 1
fi

# Clean up files.
rm -rf "${SAVED_MODEL_DIR}" "${OUTPUT_DIR}"

# 2. Test keras HDF5 --> tfjs_layers_model conversion.
KERAS_H5_PATH="$(mktemp).h5"
echo "Genearting Keras HDF5 model for testing..."
"${GENERATE_BIN}" "${KERAS_H5_PATH}" --model_type tf_keras_h5
echo "Done genearting Keras HDF5 model for testing at ${KERAS_H5_PATH}"

OUTPUT_H5_PATH="${KERAS_H5_PATH}_converted.h5"
"${CONVERTER_BIN}" \
    --input_format keras \
    --output_format tfjs_layers_model \
    "${KERAS_H5_PATH}" \
    "${OUTPUT_H5_PATH}"

if [[ ! -d "${OUTPUT_H5_PATH}" ]]; then
  echo "ERROR: Failed to find conversion output directory: ${OUTPUT_H5_PATH}" 1>&2
  exit 1
fi

# Clean up files.
rm -rf "${KERAS_H5_PATH}" "${OUTPUT_H5_PATH}"
