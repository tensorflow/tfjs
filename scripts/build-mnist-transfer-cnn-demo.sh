#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Builds the MNIST demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-mnist-demo.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/mnist_transfer_cnn_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_EPOCHS=5
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
ARTIFACTS_DIR="${DEMO_PATH}/mnist_transfer_cnn"
mkdir -p "${DEMO_PATH}"
rm -rf "${ARTIFACTS_DIR}"
mkdir -p "${ARTIFACTS_DIR}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.
export PYTHONPATH="${SCRIPTS_DIR}/..:${SCRIPTS_DIR}/../node_modules/deeplearn-src/scripts:${PYTHONPATH}"
python "${SCRIPTS_DIR}/mnist_transfer_cnn.py" \
    --epochs "${TRAIN_EPOCHS}" \
    --artifacts_dir "${ARTIFACTS_DIR}" \
    --gte5_data_path_prefix "${ARTIFACTS_DIR}/gte5" \
    --gte5_cutoff 1024

echo
echo "------------------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}/demos/mnist_transfer_cnn_demo.html"
echo "------------------------------------------------------------------"
echo

cd "${SCRIPTS_DIR}/.."
node_modules/http-server/bin/http-server -p "${DEMO_PORT}"
