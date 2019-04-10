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

cd ${SCRIPT_DIR}

yarn

# Download the tfjs repositories, build them, and link them.
if [[ ! -d "tfjs-core" ]]; then
  echo 'Use latest version of tfjs-core'
  git clone https://github.com/tensorflow/tfjs-core.git --depth 5
fi
cd tfjs-core
git checkout master && git pull
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
git checkout master && git pull
HASH_LAYERS=`git rev-parse HEAD`
# TODO(cais): This should ideally call:
#   yarn yalc link '@tensorflow/tfjs-core'
# so that tfjs-layers can be built against the HEAD of tfjs-core.
# But this doesn't work in general because the two repos frequently
# go out of sync, causing build-time and run-time errors. So right
# now we are just using the version of tfjs-core that tfjs-layers
# depends on at HEAD. The same applies to tfjs-converter and tfjs-data
# below.
rm -rf dist/ node_modules/ && yarn
yarn build && rollup -c && yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-layers'

if [[ ! -d "tfjs-converter" ]]; then
  echo 'Use latest version of tfjs-converter'
  git clone https://github.com/tensorflow/tfjs-converter.git --depth 5
fi
cd tfjs-converter
git checkout master && git pull
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
git checkout master && git pull
HASH_DATA=`git rev-parse HEAD`
rm -rf dist/ && yarn && yarn build && yalc publish

cd ..
yarn yalc link '@tensorflow/tfjs-data'

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

DATA_ROOT="${SCRIPT_DIR}/data"

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
  python "${SCRIPT_DIR}/python/benchmarks.py" "${DATA_ROOT}" \
      --hash_converter="${HASH_CONVERTER}" \
      --hash_core="${HASH_CORE}" \
      --hash_data="${HASH_DATA}" \
      --hash_layers="${HASH_LAYERS}"
fi

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo "Cleaning up virtualenv directory ${VENV_DIR}..."
  deactivate
  rm -rf "${VENV_DIR}"
fi

# Clean up virtualenv directory.
rm -rf "${VENV_DIR}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Cannot find data root directory: ${DATA_ROOT}"
  exit 1
fi

echo "Starting benchmark karma tests..."
yarn karma start karma.conf.layers.js \
    --hashes="{\"tfjs-core\":\"${HASH_CORE}\",\"tfjs-layers\":\"${HASH_LAYERS}\",\"tfjs-converter\":\"${HASH_CONVERTER}\",\"tfjs-data\": \"${HASH_DATA}\"}"
