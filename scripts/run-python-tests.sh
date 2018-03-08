#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# A script that runs all Python unit tests in tfjs-layers.

set -e

pip install -r ./scripts/requirements.txt
python -m unittest discover scripts "*_test.py"
