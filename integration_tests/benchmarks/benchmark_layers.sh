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

SKIP_PY_BENCHMAKRS=0
IS_TFJS_NODE=0  # Whether tfjs-node or tfjs-node-gpu is being benchmarked.
IS_TFJS_NODE_GPU=0  # Whether tfjs-node-gpu is being benchmarked.
LOG_FLAG=""
while true; do
  if [[ "$1" == "--skip_py_benchmarks" ]]; then
    SKIP_PY_BENCHMAKRS=1
    shift
  elif [[ "$1" == "--tfjs-node" ]]; then
    IS_TFJS_NODE=1
    IS_TFJS_NODE_GPU=0
    shift
  elif [[ "$1" == "--tfjs-node-gpu" ]]; then
    IS_TFJS_NODE=1
    IS_TFJS_NODE_GPU=1
    shift
  elif [[ "$1" == "--log" ]]; then
    LOG_FLAG="--log"
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
yarn upgrade \
    @tensorflow/tfjs-core \
    @tensorflow/tfjs-converter \
    @tensorflow/tfjs-layers \
    @tensorflow/tfjs-data \
    @tensorflow/tfjs

if [[ "${IS_TFJS_NODE}" == "1" ]]; then
  npm install -g node-gyp
  if [[ ! -d "tfjs-node" ]]; then
    echo 'Use latest version of tfjs-node'
    git clone https://github.com/tensorflow/tfjs-node.git --depth 5
  fi
  cd tfjs-node
  HASH_NODE="$(git rev-parse HEAD)"
  rm -rf dist/
  if [[ "${IS_TFJS_NODE_GPU}" == "1" ]]; then
    yarn node scripts/install.js gpu download
  else
    yarn node scripts/install.js cpu download
  fi
  yarn && yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-node'
  rm -rf .yalc/@tensorflow/tfjs-node/build
  cp -r tfjs-node/build/Release .yalc/@tensorflow/tfjs-node/build
else
  # Download the tfjs repositories, build them, and link them.
  if [[ ! -d "tfjs-core" ]]; then
    echo 'Use latest version of tfjs-core'
    git clone https://github.com/tensorflow/tfjs-core.git --depth 5
  fi
  cd tfjs-core
  HASH_CORE="$(git rev-parse HEAD)"
  rm -rf dist/ node_modules/ && yarn
  yarn build && yarn yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-core'

  if [[ ! -d "tfjs-layers" ]]; then
    echo 'Use latest version of tfjs-layers'
    git clone https://github.com/tensorflow/tfjs-layers.git --depth 5
  fi
  cd tfjs-layers
  HASH_LAYERS="$(git rev-parse HEAD)"
  # TODO(cais): This should ideally call:
  #   yarn yalc link '@tensorflow/tfjs-core'
  # so that tfjs-layers can be built against the HEAD of tfjs-core.
  # But this doesn't work in general because the two repos frequently
  # go out of sync, causing build-time and run-time errors. So right
  # now we are just using the version of tfjs-core that tfjs-layers
  # depends on at HEAD. The same applies to tfjs-converter and tfjs-data
  # below.
  rm -rf dist/ node_modules/ && yarn
  yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-layers'

  if [[ ! -d "tfjs-converter" ]]; then
    echo 'Use latest version of tfjs-converter'
    git clone https://github.com/tensorflow/tfjs-converter.git --depth 5
  fi
  cd tfjs-converter
  HASH_CONVERTER="$(git rev-parse HEAD)"
  rm -rf dist/ node_modules/ && yarn
  yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-converter'

  if [[ ! -d "tfjs-data" ]]; then
    echo 'Use latest version of tfjs-data'
    git clone https://github.com/tensorflow/tfjs-data.git --depth 5
  fi
  cd tfjs-data
  HASH_DATA="$(git rev-parse HEAD)"
  rm -rf dist/ && yarn && yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-data'
fi

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

DATA_ROOT="${SCRIPT_DIR}/data"

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo "Installing virtualenv..."
  pip install virtualenv

  VENV_DIR="$(mktemp -d)_venv"
  echo "Creating virtualenv at ${VENV_DIR} ..."
  virtualenv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"

  echo "Installing Python dependencies..."
  if [[ "${IS_TFJS_NODE_GPU}" == "1" ]]; then
    pip install -r python/requirements_gpu.txt
  else
    pip install -r python/requirements.txt
  fi

  echo "Running Python Keras benchmarks..."
  python "${SCRIPT_DIR}/python/benchmarks.py" "${DATA_ROOT}"
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

if [[ "${IS_TFJS_NODE}" == "1" ]]; then
  GPU_FLAG=""
  if [[ "${IS_TFJS_NODE_GPU}" == "1" ]]; then
    GPU_FLAG="--gpu"
  fi

  echo "Starting benchmark karma tests in Node.js..."
  yarn ts-node run_node_tests.ts \
      ${GPU_FLAG} \
      ${LOG_FLAG} \
      --hashes "{\"tfjs-node\": \"${HASH_NODE}\"}"
else
  echo "Starting benchmark karma tests in the browser..."
  yarn karma start karma.conf.layers.js \
      "${LOG_FLAG}" \
      --hashes="{\"tfjs-core\":\"${HASH_CORE}\",\"tfjs-layers\":\"${HASH_LAYERS}\",\"tfjs-converter\":\"${HASH_CONVERTER}\",\"tfjs-data\": \"${HASH_DATA}\"}"
fi
