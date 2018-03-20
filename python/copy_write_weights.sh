#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Copy the write weights file from deeplearnjs-src to the Python source tree.
#
# This is a required step before you run the Python unit tests or build the
# tensorflowjs pip package.

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

yarn

cp "${SCRIPTS_DIR}/../node_modules/deeplearn-src/scripts/write_weights.py" \
  "${SCRIPTS_DIR}/tensorflowjs/"
