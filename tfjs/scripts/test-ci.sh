#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
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

# Exit immediately if a command exits with a non-zero status.
set -e

node ../scripts/run_flaky.js "yarn karma start --browsers='bs_firefox_mac' --singleRun"
node ../scripts/run_flaky.js "yarn karma start --browsers='bs_chrome_mac' --singleRun"
yarn test-tools

# If these are re-enabled, note that they may run multiple browsers per karma
# instance.
# cd integration_tests
# yarn benchmark-cloud
# Reinstall the following line once https://github.com/tensorflow/tfjs/pull/1663
# is resolved.
# yarn benchmark --layers --tfjs-node
# TODO(lina) update this to work against head.
# yarn validate-converter --tfjs-node
# cd ../../
