#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Builds the benchmarks demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-benchmarks-demo.sh
#
# Then open the demo HTML page in your browser, e.g.,
#   google-chrome demos/benchmarks_demo.html

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build TensorFlow.js standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

DEMO_PATH="${SCRIPTS_DIR}/../dist/demo"
mkdir -p "${DEMO_PATH}"

# Run Python script to generate the model and weights JSON files.
# The extension names are ".js" because they will later be converted into
# sourceable JavaScript files.
PYTHONPATH="${SCRIPTS_DIR}/.." python \
  "${SCRIPTS_DIR}/benchmarks.py" "${DEMO_PATH}/benchmarks.keras.js"

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/benchmarks_demo.html &"

