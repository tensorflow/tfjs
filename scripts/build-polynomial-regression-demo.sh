#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Builds the Polynomial Regression Demo for TensorFlow.js Layers.
# Usage example: do under the root of the source repository:
#   ./scripts/build-polynomial-regression-demo.sh

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build TensorFlow.js Layers standalone.
"${SCRIPTS_DIR}/build-standalone.sh"

echo
echo "Now you can open the demo by:"
echo "  google-chrome demos/polynomial_regression_demo.html &"
