#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

TEST_DATA="test-data/"

yarn link
cd integration_tests/tfjs2keras/
yarn
yarn link @tensorflow/tfjs-layers
rm -rf "$TEST_DATA"
mkdir "$TEST_DATA"
node tfjs_save.js "$TEST_DATA"
cd ../..
