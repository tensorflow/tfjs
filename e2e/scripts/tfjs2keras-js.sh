#!/usr/bin/env bash

# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

TEST_DATA="integration_tests/create_save_predict_data/"

rm -rf "$TEST_DATA"
mkdir "$TEST_DATA"

node integration_tests/tfjs_save.js "$TEST_DATA"
