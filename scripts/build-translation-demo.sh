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

if [[ ! -f ${DATA_PATH} ]]; then
  echo "ERORR: Cannot find training data at path '${DATA_PATH}'"
  exit 1
fi

TRAIN_EPOCHS=100
DEMO_PORT=8000
while true; do
  if [[ "$1" == "--port" ]]; then
    DEMO_PORT=$2
    shift 2
  elif [[ "$1" == "--epochs" ]]; then
    TRAIN_EPOCHS=$2
    shift 2
  elif [[ -z "$1" ]]; then
    break
  else
    echo "ERROR: Unrecognized argument: $1"
    exit 1
  fi
done

# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
ARTIFACTS_DIR="${DEMO_PATH}/translation"
mkdir -p "${DEMO_PATH}"
rm -rf "${ARTIFACTS_DIR}"

# Train the model and generate:
#   * saved model in TensorFlow.js
#   * metadata JSON file.
export PYTHONPATH="${SCRIPTS_DIR}/..:${SCRIPTS_DIR}/../node_modules/deeplearn-src/scripts:${PYTHONPATH}"
python "${SCRIPTS_DIR}/translation.py" \
    "${DATA_PATH}" \
    --recurrent_initializer glorot_uniform \
    --artifacts_dir "${ARTIFACTS_DIR}" \
    --epochs "${TRAIN_EPOCHS}"
# TODO(cais): This --recurrent_initializer is a workaround for the limitation
# in TensorFlow.js Layers that the default recurrent initializer "Orthogonal" is
# currently not supported. Remove this once "Orthogonal" becomes available.

echo
echo "-----------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}/demos/translation_demo.html"
echo "-----------------------------------------------------------"
echo

node_modules/http-server/bin/http-server -p "${DEMO_PORT}"
