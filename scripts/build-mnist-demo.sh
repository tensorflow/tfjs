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
#   google-chrome demos/mnist_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse input arguments.
NO_TRAIN=0
while true; do
  if [[ "$1" == "--no_train" ]]; then
    NO_TRAIN=1
  elif [[ -z "$1" ]]; then
    break
  fi
  shift
done

# Build TensorFlow.js standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

if [[ "${NO_TRAIN}" != "1" ]]; then
  # Run Python script to generate the model and weights JSON files.
  # The extension names are ".js" because they will later be converted into
  # sourceable JavaScript files.
  PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/mnist.py" \
      --model_json_path "${DEMO_PATH}/mnist.keras.model.json" \
      --weights_json_path "${DEMO_PATH}/mnist.keras.weights.json"

  # Prepend "const * = " to the json files
  printf "const mnistModelJSON = " > "${DEMO_PATH}/mnist.keras.model.js"
  cat "${DEMO_PATH}/mnist.keras.model.json" >> "${DEMO_PATH}/mnist.keras.model.js"
  printf ";" >> "${DEMO_PATH}/mnist.keras.model.js"
  rm "${DEMO_PATH}/mnist.keras.model.json"

  printf "const mnistWeightsJSON = " > "${DEMO_PATH}/mnist.keras.weights.js"
  cat "${DEMO_PATH}/mnist.keras.weights.json" >> "${DEMO_PATH}/mnist.keras.weights.js"
  printf ";" >> "${DEMO_PATH}/mnist.keras.weights.js"
  rm "${DEMO_PATH}/mnist.keras.weights.json"
else
  printf "\nSkipped model training.\n"
fi

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/mnist_demo.html &"

