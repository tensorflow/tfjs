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

yarn lint
# Test in node (headless environment).
yarn test-node-ci

if [ "$NIGHTLY" = true ]
then
  # Run the first karma separately so it can download the BrowserStack binary
  # without conflicting with others.
  yarn run-browserstack --browsers=bs_safari_mac,bs_ios_11 --testEnv webgl1 --flags '{"WEBGL_CPU_FORWARD": false, "WEBGL_SIZE_UPLOAD_UNIFORM": 0}'

  # Run the rest of the karma tests in parallel. These runs will reuse the
  # already downloaded binary.
  npm-run-all -p -c --aggregate-output \
    "run-browserstack --browsers=bs_safari_mac,bs_ios_11,bs_android_9 --flags '{\"HAS_WEBGL\": false}' --testEnv cpu" \
    "run-browserstack --browsers=bs_firefox_mac,bs_chrome_mac" \
    "run-browserstack --browsers=bs_chrome_mac,win_10_chrome,bs_android_9 --testEnv webgl2 --flags '{\"WEBGL_CPU_FORWARD\": false, \"WEBGL_SIZE_UPLOAD_UNIFORM\": 0}'" \
    "run-browserstack --browsers=bs_chrome_mac --testEnv webgl2 --flags '{\"WEBGL_PACK\": false}'" \

  ### The next section tests TF.js in a webworker.
  # Make a dist/tf-core.min.js file to be imported by the web worker.
  yarn rollup -c --ci
  # Safari doesn't have offscreen canvas so test cpu in a webworker.
  # Chrome has offscreen canvas, so test webgl in a webworker.
  yarn test-webworker --browsers=bs_safari_mac,bs_chrome_mac
else
  yarn run-browserstack --browsers=bs_chrome_mac
fi
