#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Builds the IMDB demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-imdb-demo.sh lstm
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/imdb_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  build-imdb-demo.sh <MODEL_TYPE>"
  echo
  echo "MODEL_TYPE options: lstm | cnn"
  exit 1
fi
MODEL_TYPE=$1
shift
echo "Using model type: ${MODEL_TYPE}"

DEMO_PORT=8000
TRAIN_EPOCHS=5
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
mkdir -p "${DEMO_PATH}"

# Make sure you install the tensorflowjs pip package first.

PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/imdb.py" \
    "${MODEL_TYPE}" \
    --epochs "${TRAIN_EPOCHS}" \
    --artifacts_dir "${DEMO_PATH}/imdb"

echo
echo "-----------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}/demos/imdb_demo.html"
echo "-----------------------------------------------------------"
echo

cd "${SCRIPTS_DIR}/.."
node_modules/http-server/bin/http-server -p "${DEMO_PORT}"
