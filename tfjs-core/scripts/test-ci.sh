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

# Test in node (headless environment).
yarn test-node-ci

if [ "$NIGHTLY" = true ]
then
  # TODO(mattsoulanille): Is the following comment still relevant now that we're
  # runing each browser separately?
  # Run the first karma separately so it can download the BrowserStack binary
  # without conflicting with others.
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_chrome_mac"

  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_firefox_mac --flags '{"\""HAS_WEBGL"\"": false}' --testEnv cpu"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_safari_mac --flags '{"\""HAS_WEBGL"\"": false}' --testEnv cpu"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_ios_11 --flags '{"\""HAS_WEBGL"\"": false}' --testEnv cpu"


  ### The next section tests TF.js in a webworker using the CPU backend.
  echo "Start webworker test."
  # Make a dist/tf-core.min.js file to be imported by the web worker.
  yarn rollup -c --ci
  # copy the cpu backend bundle somewhere the test can access it
  cp -v ../tfjs-backend-cpu/dist/tf-backend-cpu.min.js dist/
  node ../scripts/run_flaky.js "yarn test-webworker --browsers=bs_safari_mac"
  node ../scripts/run_flaky.js "yarn test-webworker --browsers=bs_chrome_mac"
else
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_chrome_mac"
fi
