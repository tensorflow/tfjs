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
  rm -rf tfjs-node/ tfjs-node-gpu
  cp -r ../../tfjs-node ../../tfjs-node-gpu .

  if [[ "${IS_TFJS_NODE_GPU}" == "1" ]]; then
    pushd tfjs-node-gpu
  else
    # The Python library for comparison may contain both the CPU and GPU
    # kernels. The line below prevent the GPU from being used during CPU
    # benchmarks.
    export CUDA_VISIBLE_DEVICES=""
    pushd tfjs-node
  fi

  HASH_NODE="$(git rev-parse HEAD)"
  rm -rf dist/
  yarn
  rm -f tensorflow-tfjs-*.tgz

  yarn build-npm

  TAR_BALL_COUNT="$(ls tensorflow-tfjs-node-*.tgz | wc -w)"
  if [[ "${TAR_BALL_COUNT}" != "1" ]]; then
    echo "ERROR: Expected to find exactly one tensorflow-tfjs-node-* tar ball"
    echo "       But found ${TAR_BALL_COUNT}."
    exit 1
  fi

  TAR_BALL="$(find ./ -name "tensorflow-tfjs-node-*.tgz")"
  if [[ -z "${TAR_BALL}" ]]; then
    echo "Unable to find the tar ball built by `yarn build-npm`."
    exit 1
  fi
  echo "Found built tar ball at: ${TAR_BALL}"

  rm -rf ../node_modules/@tensorflow/tfjs-node/*
  tar xvzf "${TAR_BALL}" --directory ../node_modules/@tensorflow/tfjs-node
  pushd ../node_modules/@tensorflow/tfjs-node/

  mv package/* ./
  rm -rf package/

  # Compile the bindings in the tfjs-node / tfjs-node-gpu release package.
  yarn
  yarn build-addon-from-source

  popd
  popd
else
  # NOTE: Browser benchmarks are compared with Python CPU.
  export CUDA_VISIBLE_DEVICES=""

  # Copy the tfjs repositories, build them, and link them.
  rm -rf tfjs-core/
  cp -r ../../tfjs-core .

  cd tfjs-core
  HASH_CORE="$(git rev-parse HEAD)"
  rm -rf dist/ node_modules/ && yarn
  yarn build && yarn yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-core'

  rm -rf tfjs-layers/
  cp -r ../../tfjs-layers .

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

  rm -rf tfjs-converter/
  cp -r ../../tfjs-converter .

  cd tfjs-converter
  HASH_CONVERTER="$(git rev-parse HEAD)"
  rm -rf dist/ node_modules/ && yarn
  yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-converter'

  rm -rf tfjs-data/
  cp -r ../../tfjs-data .

  cd tfjs-data
  HASH_DATA="$(git rev-parse HEAD)"
  rm -rf dist/ && yarn && yarn build && yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-data'
fi

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

if [[ -z "$(which pip)" ]]; then
  echo "pip is not on path. Attempting to install it..."
  apt-get update
  apt-get install -y python-pip
fi

DATA_ROOT="${SCRIPT_DIR}/data"

if [[ "${SKIP_PY_BENCHMAKRS}" == 0 ]]; then
  echo "Installing virtualenv..."
  pip install virtualenv

  VENV_DIR="$(mktemp -d)_venv"
  echo "Creating virtualenv at ${VENV_DIR} ..."
  virtualenv -p python3 "${VENV_DIR}"
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
      --filename "models/benchmarks.ts" \
      ${GPU_FLAG} \
      ${LOG_FLAG} \
      --hashes "{\"tfjs-node\": \"${HASH_NODE}\"}"
else
  echo "Starting benchmark karma tests in the browser..."
  yarn karma start karma.conf.layers.js \
      "${LOG_FLAG}" \
      --hashes="{\"tfjs-core\":\"${HASH_CORE}\",\"tfjs-layers\":\"${HASH_LAYERS}\",\"tfjs-converter\":\"${HASH_CONVERTER}\",\"tfjs-data\": \"${HASH_DATA}\"}"
fi
