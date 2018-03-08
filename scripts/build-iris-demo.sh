#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Builds the Iris demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-iris-demo.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/iris_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.
PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/iris.py" \
    --model_json_path "${DEMO_PATH}/iris.keras.model.json" \
    --weights_json_path "${DEMO_PATH}/iris.keras.weights.json"

# Prepend "const * = " to the json files.
printf "const irisModelJSON = " > "${DEMO_PATH}/iris.keras.model.js"
cat "${DEMO_PATH}/iris.keras.model.json" >> "${DEMO_PATH}/iris.keras.model.js"
printf ";" >> "${DEMO_PATH}/iris.keras.model.js"
rm "${DEMO_PATH}/iris.keras.model.json"

printf "const irisWeightsJSON = " > "${DEMO_PATH}/iris.keras.weights.js"
cat "${DEMO_PATH}/iris.keras.weights.json" >> "${DEMO_PATH}/iris.keras.weights.js"
printf ";" >> "${DEMO_PATH}/iris.keras.weights.js"
rm "${DEMO_PATH}/iris.keras.weights.json"

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/iris_demo.html &"

