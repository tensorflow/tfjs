#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# A script that runs all Python unit tests in tfjs-layers.

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPTS_DIR}/copy_write_weights.sh"

TEST_FILES_GLOB='**/*_test.py'
if [[ ! -z $1 ]]; then
  TEST_FILES_GLOB="$1"
  echo "TEST_FILES_GLOB: ${TEST_FILES_GLOB}"
fi

pip install -r "${SCRIPTS_DIR}/requirements.txt"

cd "${SCRIPTS_DIR}"

TEST_FILES=tensorflowjs/${TEST_FILES_GLOB}

export PYTHONPATH=".:${PYTHONPATH}"
for TEST_FILE in ${TEST_FILES}; do
  echo "Running test: ${TEST_FILE}"
  python "${TEST_FILE}"
done

echo
echo "All tests passed."
echo
