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

yarn build
yarn lint
yarn ts-node ./scripts/release_notes/run_tests.ts
yarn karma start --browsers='bs_firefox_mac,bs_chrome_mac' --singleRun
cd integration_tests
yarn benchmark-cloud
cd ../../
