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
#   benchmarks.sh
#     This runs the core benchmarks.
#   benchmarks.sh --layers
#     This runs the layers benchmarks in the browser.
#   benchmakrs.sh --node
#     This runs tfjs-node benchmarks (TODO(cais): Implement it.)

set -e

SUITE="core"
OTHER_ARGS=""
for ARG in "$@"; do
  if [[ "${ARG}" == "--layers" ]]; then
    SUITE="layers"
  else
    OTHER_ARGS="${OTHER_ARGS} ${ARG} "
  fi
done

echo "SUITE: ${SUITE}"

if [[ "${SUITE}" == "core" ]]; then
  karma start --firebaseKey "${FIREBASE_KEY}"
elif [[ "${SUITE}" == "layers" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  "${SCRIPT_DIR}/benchmark_layers.sh" ${OTHER_ARGS}
elif [[ "${SUITE}" == "node" ]]; then
  echo "ERROR: node benchmark suite is not implemented yet."
  exit 1
else
  echo "ERROR: Unrecognized suite name: ${SUITE}"
  exit 1
fi
