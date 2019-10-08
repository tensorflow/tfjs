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

# Exit all child processes when this script terminates.
# While we will exit child processes on successful completion, any early exit
# after a child process is created will be handled by this trap.
trap cleanup INT TERM ERR

cleanup() {
  echo "Clean up metro $metro_pid"
  kill -9 $metro_pid
}

yarn
yarn lint

# Copy the built version of tfjs-core from the current checkout
if [ "$1" == "skip-core-build" ]; then
  echo "Skip core build"
  cd ../../tfjs-core && cp -rf dist ../tfjs-react-native/integration_rn59/node_modules/@tensorflow/tfjs-core && cd ../tfjs-react-native/integration_rn59
else
  cd ../../tfjs-core && yarn && yarn build-ci && cp -rf dist ../tfjs-react-native/integration_rn59/node_modules/@tensorflow/tfjs-core && cd ../tfjs-react-native/integration_rn59
fi

yarn prep-tests

# Spawn a metro bundler/asset server in the background.
nohup yarn start-metro &
let metro_pid=$!
echo "Started metro. PID=$metro_pid"

# Start the test suite.
yarn test-integration
test_result=$?
echo "Today is $test_result"

# Kill the child process explicitly so that the exit code of the script
# isn't changed by it's eventual termination.
kill -TERM $metro_pid
# wait $metro_pid
echo $metro_pid was terminated.

# Return the exit code of the test
exit $test_result
