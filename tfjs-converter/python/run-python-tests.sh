#!/usr/bin/env bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# A script that runs all Python unit tests in tfjs-layers.

function print_usage() {
  echo "Usage:"
  echo "  run-python-tests.sh <requirments_file>"
  echo
}

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_FILES="$(find "${SCRIPTS_DIR}" -name '*_test.py')"

pip install virtualenv

TMP_VENV_DIR="$(mktemp -u).venv"
virtualenv -p "python" "${TMP_VENV_DIR}"
source "${TMP_VENV_DIR}/bin/activate"

# There is one argument (requirements_file), please update this constant when
# you adding more arguments.
ARGS_COUNT=1

# Default requirements file name.
REQ_FILE="${SCRIPTS_DIR}/requirements-dev.txt"

# Show the usage message if there are too many arguments.
if [[ $# > ARGS_COUNT ]]; then
  print_usage
  exit 1
fi

# Use the user specified requirements file name.
if [[ $# == 1 ]]; then
  REQ_FILE=$1
fi
pip install -r "${REQ_FILE}"

# Run pylint for tensorflowjs directory
cd "${SCRIPTS_DIR}"
pylint --rcfile=.pylintrc tensorflowjs

export PYTHONPATH=".:${PYTHONPATH}"
for TEST_FILE in ${TEST_FILES}; do
  echo
  echo "====== Running test: ${TEST_FILE} ======"
  echo
  python "${TEST_FILE}"
done

echo
echo "All tests passed."
echo

# Clean up
deactivate
rm -rf "${TMP_VENV_DIR}"
