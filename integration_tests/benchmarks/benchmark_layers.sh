#!/usr/bin/env bash
#
# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
#
# Builds the benchmarks demo for TensorFlow.js Layers.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PORT=8090
SKIP_PY_BENCHMAKRS=0
while true; do
  if [[ "$1" == "--port" ]]; then
    DATA_PORT=$2
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

DATA_ROOT="${SCRIPT_DIR}/data"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo "Installing virtualenv..."
  pip install virtualenv

  VENV_DIR="$(mktemp -d)"
  echo "Creating virtualenv at ${VENV_DIR} ..."
  virtualenv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"

  echo "Installing Python dependencies..."
  pip install -r python/requirements.txt

  echo "Running Python Keras benchmarks..."
  python "${SCRIPT_DIR}/python/benchmarks.py" "${DATA_ROOT}"
fi

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo "Cleaning up virtualenv directory ${VENV_DIR}..."
  deactivate
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Cannot find data root directory: ${DATA_ROOT}"
  exit 1
fi

cd ${SCRIPT_DIR}

yarn

# Download the tfjs repositories, build them, and link them.
if [[ ! -d "tfjs-core" ]]; then
  echo 'Use latest version of tfjs-core'
  git clone https://github.com/tensorflow/tfjs-core.git --depth 5
fi
cd tfjs-core
HASH_CORE=`git rev-parse HEAD`
rm -rf dist/ node_modules/ && yarn
yarn build && yarn yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-core'

if [[ ! -d "tfjs-layers" ]]; then
  echo 'Use latest version of tfjs-layers'
  git clone https://github.com/tensorflow/tfjs-layers.git --depth 5
fi
cd tfjs-layers
HASH_LAYERS=`git rev-parse HEAD`
rm -rf dist/ node_modules/ && yarn
# yarn yalc link '@tensorflow/tfjs-core'
yarn build && rollup -c && yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-layers'

if [[ ! -d "tfjs-converter" ]]; then
  echo 'Use latest version of tfjs-converter'
  git clone https://github.com/tensorflow/tfjs-converter.git --depth 5
fi
cd tfjs-converter
HASH_CONVERTER=`git rev-parse HEAD`
rm -rf dist/ node_modules/ && yarn
yarn build && yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-converter'

if [[ ! -d "tfjs-data" ]]; then
  echo 'Use latest version of tfjs-data'
  git clone https://github.com/tensorflow/tfjs-data.git --depth 5
fi
cd tfjs-data
HASH_DATA=`git rev-parse HEAD`
rm -rf dist/ && yarn && yarn build && yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-data'

echo "Starting benchmark tests..."
yarn karma start karma.conf.layers.js

# echo
# echo "-----------------------------------------------------------"
# echo "Benchmarks page will open in your browser shortly...       "
# echo
# echo "Once the page is up, click the 'Run Benchmarks' button     "
# echo "to run the benchmarks in the browser.                      "
# echo "-----------------------------------------------------------"
# echo
