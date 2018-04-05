#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Builds the benchmarks demo for TensorFlow.js Layers.
# Usage example: do this from the 'benchmarks' directory:
#   ./build-benchmarks.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   http://localhost:8000/dist

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEMO_PORT=8000
while true; do
  if [[ "$1" == "--port" ]]; then
    DEMO_PORT=$2
    shift 2
  elif [[ -z "$1" ]]; then
    break
  else
    echo "ERROR: Unrecognized argument: $1"
    exit 1
  fi
done

DATA_ROOT="${DEMO_DIR}/dist/data"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

# Make sure you install the tensorflowjs pip package first.

echo Running Python Keras benchmarks...
python "${DEMO_DIR}/python/benchmarks.py" "${DATA_ROOT}"

echo Building local TensorFlow.js Layers NPM package...
cd ../..
yarn build
yarn link

cd ${DEMO_DIR}
yarn
yarn link "@tensorflow/tfjs-layers"
yarn build

echo
echo "-----------------------------------------------------------"
echo "Once the HTTP server has started, you can view the demo at:"
echo "  http://localhost:${DEMO_PORT}/dist"
echo "-----------------------------------------------------------"
echo

node_modules/http-server/bin/http-server -p "${DEMO_PORT}"
