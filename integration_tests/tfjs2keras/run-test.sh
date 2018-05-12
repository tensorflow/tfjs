#!/bin/bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

if [[ -z "${TRAVIS_BUILD_NUMBER}" ]]; then
  pip install -r requirements.txt
else
  # If in Travis, use the `--user` flag when performing `pip install` of
  # dependencies.
  pip install --user -r requirements.txt
fi

python tfjs2keras_test.py
