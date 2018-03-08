#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Builds the sequence-to-sequence English-French translation demo for
# TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-translation-demo.sh ~/ml-data/fra-eng/fra.txt
#
# You can specify the number of training epochs by using the --epochs flag.
# For example:
#   ./scripts/build-translation-demo.sh ~/ml-data/fra-eng/fra.txt --epochs 10
#
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/translation_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="$1"
if [[ -z "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH is not specified."
  echo "You can download the training data with a command such as:"
  echo "  wget http://www.manythings.org/anki/fra-eng.zip"
  exit 1
fi
shift 1

EPOCHS_FLAG=""
while true; do
  if [[ ! -z "$1" ]]; then
    if [[ "$1" == "--epochs" ]]; then
      EPOCHS_FLAG="--epochs $2"
      shift 2
    else
      echo "ERROR: Unrecognized flag: $1"
      exit 1
    fi
  else
    break
  fi
done

# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

# TODO(cais): Do not hardcode the paths. Obtain them from a Python training
#   script.
MODEL_JSON="${DEMO_PATH}/translation.keras.model.json"
WEIGHTS_JSON="${DEMO_PATH}/translation.keras.weights.json"
METADATA_JSON="${DEMO_PATH}/translation.keras.metadata.json"

# Train the model and generate:
#   * model JSON file
#   * weights JSON file
#   * metadata JSON file.
PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/translation.py" \
    "${DATA_PATH}" \
    --recurrent_initializer glorot_uniform \
    --model_json_path "${MODEL_JSON}" \
    --weights_json_path "${WEIGHTS_JSON}" \
    --metadata_json_path "${METADATA_JSON}" \
    ${EPOCHS_FLAG}
# TODO(cais): This --recurrent_initializer is a workaround for the limitation
# in TensorFlow.js Layers that the default recurrent initializer "Orthogonal" is
# currently not supported. Remove this once "Orthogonal" becomes available.

# Prepend "const * = " to the json files.
MODEL_JS="${DEMO_PATH}/translation.keras.model.js"
printf "const translationModelJSON = " > "${MODEL_JS}"
cat "${MODEL_JSON}" >> "${MODEL_JS}"
printf ";" >> "${MODEL_JS}"
rm "${MODEL_JSON}"

# Prepend "const * = " to metadata (includes token indices etc.).
METADATA_JS="${DEMO_PATH}/translation.keras.metadata.js"
printf "const translationMetadata = " > "${METADATA_JS}"
cat "${METADATA_JSON}" >> "${METADATA_JS}"
printf ";" >> "${METADATA_JS}"
rm "${METADATA_JSON}"

WEIGHTS_JS=""${DEMO_PATH}/translation.keras.weights.js""
printf "const translationWeightsJSON = " > "${WEIGHTS_JS}"
cat "${WEIGHTS_JSON}" >> "${WEIGHTS_JS}"
printf ";" >> "${WEIGHTS_JS}"
rm "${WEIGHTS_JSON}"

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/translation_demo.html &"
