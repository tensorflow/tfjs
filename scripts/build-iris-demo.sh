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

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
ARTIFACTS_DIR="${DEMO_PATH}/iris"
rm -rf "${ARTIFACTS_DIR}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.

# Make sure you install the tensorflowjs pip package first.

export PYTHONPATH="${SCRIPTS_DIR}/.."
python "${SCRIPTS_DIR}/iris.py" --artifacts_dir "${ARTIFACTS_DIR}"

echo
echo "----------------------------------------------------------------"
echo "TensorFlow.js artifacts have been generated at"
echo "  ${ARTIFACTS_DIR}"
echo
echo "See https://github.com/tensorflow/tfjs-examples/tree/master/iris"
echo "for how to load and serve the saved model in the browser."
echo "----------------------------------------------------------------"
echo
