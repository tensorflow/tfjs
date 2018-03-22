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

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_FILES="$(find "${SCRIPTS_DIR}" -name '*_test.py')"

pip install -r "${SCRIPTS_DIR}/requirements.txt"

cd "${SCRIPTS_DIR}"

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
