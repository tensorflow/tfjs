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
SKIP_PY_BENCHMAKRS=0
while true; do
  if [[ "$1" == "--port" ]]; then
    DEMO_PORT=$2
    shift 2
  elif [[ "$1" == "--skip_py_benchmarks" ]]; then
    SKIP_PY_BENCHMAKRS=1
    shift
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

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo Running Python Keras benchmarks...
  python "${DEMO_DIR}/python/benchmarks.py" "${DATA_ROOT}"
fi

# Build tfjs-layers.
echo "Building tfjs-layers..."
cd "${DEMO_DIR}/../.."
yarn
yarn build

echo "Building benchmarks demo..."
cd ${DEMO_DIR}
# Force a refresh of the tfjs-layers dependency.
rm -rf node_modules
yarn
yarn build

echo
echo "-----------------------------------------------------------"
echo "Benchmarks page will open in your browser shortly...       "
echo
echo "Once the page is up, click the 'Run Benchmarks' button     "
echo "to run the benchmarks in the browser.                      "
echo "-----------------------------------------------------------"
echo

yarn watch
