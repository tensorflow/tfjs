#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# A script that runs all Python unit tests in tfjs-layers.

set -e

TEST_FILES_GLOB='*_test.py'
if [[ ! -z $1 ]]; then
  TEST_FILES_GLOB="$1"
  echo "TEST_FILES_GLOB: ${TEST_FILES_GLOB}"
fi

pip install -r ./scripts/requirements.txt

TEST_FILES=scripts/${TEST_FILES_GLOB}

export PYTHONPATH="./:./node_modules/deeplearn-src/scripts:$PYTHONPATH"
for TEST_FILE in ${TEST_FILES}; do
  BASE_FILE_NAME="$(realpath --relative-to="scripts" ${TEST_FILE})"
  python -m unittest discover scripts "${BASE_FILE_NAME}"
done

echo
echo "All tests passed."
echo
