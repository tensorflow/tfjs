#!/usr/bin/env bash
#
# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_PY_BENCHMAKRS=0
IS_TFJS_NODE=0  # Whether tfjs-node or tfjs-node-gpu is being benchmarked.
IS_TFJS_NODE_GPU=0  # Whether tfjs-node-gpu is being benchmarked.
LOG_FLAG=""
while true; do
  if [[ "$1" == "--tfjs-node" ]]; then
    IS_TFJS_NODE=1
    IS_TFJS_NODE_GPU=0
    shift
  elif [[ "$1" == "--tfjs-node-gpu" ]]; then
    IS_TFJS_NODE=1
    IS_TFJS_NODE_GPU=1
    shift
  elif [[ "$1" == "--log" ]]; then
    HASH="$(git rev-parse HEAD)"

    if [[ -z "${HASH}" ]]; then
      echo "ERROR: Failed to retrieve git commit hash" 1>&2
      exit 1
    fi

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
    @tensorflow/tfjs

if [[ "${IS_TFJS_NODE}" == "1" ]]; then
  rm -rf tfjs-node/
  cp -r ../../tfjs-node .

  cd tfjs-node
  rm -rf dist/
  if [[ "${IS_TFJS_NODE_GPU}" == "1" ]]; then
    yarn node scripts/install.js gpu download
  else
    yarn node scripts/install.js cpu download
  fi
  yarn && yarn build && yarn yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-node'
  # yalc publish does not deliver libtensorflow and node native addon, so we
  # need to copy libtensorflow and build addon from source
  cp -r tfjs-node/deps .yalc/@tensorflow/tfjs-node/
  cd .yalc/@tensorflow/tfjs-node
  yarn && yarn build-addon-from-source
  cd ../../..
else
  rm -rf tfjs-core/
  cp -r ../../tfjs-core .

  cd tfjs-core
  rm -rf dist/ node_modules/ && yarn
  yarn build && yarn yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-core'

  rm -rf tfjs-converter/
  cp -r ../../tfjs-converter .

  cd tfjs-converter
  rm -rf dist/ node_modules/ && yarn
  yarn build && yarn yalc publish

  cd ..
  yarn yalc link '@tensorflow/tfjs-converter'
fi

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

if [[ -z "$(which pip)" ]]; then
  echo "pip is not on path. Attempting to install it..."
  apt-get update
  apt-get install -y python3-pip python3-dev
fi

DATA_ROOT="${SCRIPT_DIR}/data"

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

echo "Running python converter..."
python "${SCRIPT_DIR}/python/validation.py" "${DATA_ROOT}"

echo "Cleaning up virtualenv directory ${VENV_DIR}..."
deactivate
rm -rf "${VENV_DIR}"

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

  echo "Starting validation karma tests in Node.js..."
  yarn ts-node run_node_tests.ts \
      --filename "models/validation.ts" \
      ${GPU_FLAG} \
      ${LOG_FLAG} \
      --hashes "{\"tfjs-node\": \"${HASH}\"}"
else
  echo "Starting validation karma tests in the browser..."
  yarn karma start karma.conf.validations.js \
      "${LOG_FLAG}" \
      --hashes="{\"tfjs-core\":\"${HASH}\",\"tfjs-converter\":\"${HASH}\"}"
fi
