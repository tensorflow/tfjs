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

set -e

# Exit all child processes when this script terminates
trap "exit" INT TERM ERR
trap "kill 0" EXIT

yarn
yarn lint

# build tfjs-core from current checkout

# copy dist folder of tfjs-core to integration_test node_modules

# Spawn a metro bundler/asset server in the background.
nohup yarn start-metro &>/dev/null &
let metro_pid=$!
echo "Started metro. PID=$metro_pid"

# Start the test suite. Capture the output of this as the final return
# value for this script.
yarn test-integration
