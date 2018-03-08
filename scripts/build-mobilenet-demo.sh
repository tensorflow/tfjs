#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Builds the MobileNet demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-mobilenet-demo.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/mobilenet_demo.html &

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.
PYTHONPATH="${SCRIPTS_DIR}/.." python "${SCRIPTS_DIR}/mobilenet.py" \
    --mode "serialize" \
    --model_json_path "${DEMO_PATH}/mobilenet.keras.model.json" \
    --weights_json_path "${DEMO_PATH}/mobilenet.keras.weights.json"

# Copy and convert class names json file to local js file.
cp "${SCRIPTS_DIR}/imagenet_class_names.json" "${DEMO_PATH}/"

echo
echo "Now you can open the demo by:"
echo "  python -m SimpleHTTPServer"
echo "  google-chrome http://localhost:8000/demos/mobilenet_demo.html &"
