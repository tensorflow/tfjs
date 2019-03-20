# @license
# Copyright 2019 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Usage:
#   benchmarks.sh <SUITE>
#
# Args:
#   SUITE Name of the benchmark suite to run: core | layers | node

set -e

SUITE=$1
if [[ -z "${SUITE}" ]]; then
  echo "ERORR: suite name is not specific. Options: core | layers | node" 2>&1
  exit 1
elif [[ "${SUITE}" == "core" ]]; then
  karma start --firebaseKey $FIREBASE_KEY
elif [[ "${SUITE}" == "layers" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  shift
  "${SCRIPT_DIR}/benchmark_layers.sh" $@
elif [[ "${SUITE}" == "node" ]]; then
  echo "ERROR: node benchmark suite is not implemented yet."
  exit 1
fi
