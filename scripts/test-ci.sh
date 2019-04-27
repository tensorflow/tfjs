#!/usr/bin/env bash
# Copyright 2018 Google LLC. All Rights Reserved.
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
# =============================================================================

set -e

yarn build-ci
yarn lint
# Test in node (headless environment).
yarn test-node-ci

# Run the first karma separately so it can download the BrowserStack binary
# without conflicting with others.
yarn run-browserstack --browsers=bs_safari_mac --backend webgl --flags '{"WEBGL_CPU_FORWARD": false, "WEBGL_SIZE_UPLOAD_UNIFORM": 0}'

# Run the rest of the karma tests in parallel. These runs will reuse the
# already downloaded binary.
npm-run-all -p -c --aggregate-output \
  "run-browserstack --browsers=bs_safari_mac --flags '{\"HAS_WEBGL\": false}' --backend cpu" \
  "run-browserstack --browsers=win_10_chrome --backend webgl --flags '{\"WEBGL_CPU_FORWARD\": false, \"WEBGL_SIZE_UPLOAD_UNIFORM\": 0}'" \
  "run-browserstack --browsers=bs_ios_11 --backend webgl --flags '{\"WEBGL_CPU_FORWARD\": false, \"WEBGL_SIZE_UPLOAD_UNIFORM\": 0}'" \
  "run-browserstack --browsers=bs_ios_11 --flags '{\"HAS_WEBGL\": false}' --backend cpu" \
  "run-browserstack --browsers=bs_firefox_mac" \
  "run-browserstack --browsers=bs_chrome_mac" \
  "run-browserstack --browsers=bs_chrome_mac --backend webgl --flags '{\"WEBGL_CPU_FORWARD\": true}'" \
  "run-browserstack --browsers=bs_chrome_mac --backend webgl --flags '{\"WEBGL_CPU_FORWARD\": false}'"
