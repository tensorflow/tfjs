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

yarn
yarn lint

# Default is to run tests against the packages specified dependencies. We can
# optionally copy a built version of tfjs-core from the current checkout
if [ "$1" == "use-core-build" ]; then
  # Assume core has been built recently, just link it in. Use cp and not
  # yalc to avoid symlinks
  echo "Use existing core build"
  cd ../../tfjs-core && cp -rf dist ../tfjs-react-native/integration_rn59/node_modules/@tensorflow/tfjs-core && cd ../tfjs-react-native/integration_rn59
elif [ "$1" == "build-head" ]; then
  # Build head and link it in.
  echo "Build head from core"
  cd ../../tfjs-core && yarn && yarn build-ci && cp -rf dist ../tfjs-react-native/integration_rn59/node_modules/@tensorflow/tfjs-core && cd ../tfjs-react-native/integration_rn59
fi

# build tfjs-react-native from head
cd ../ && yarn && yarn build && cp -rf dist ./integration_rn59/node_modules/@tensorflow/tfjs-react-native && cd integration_rn59

yarn prep-tests

# Spawn a metro bundler/asset server in the background.
yarn start-metro &
let metro_pid=$!
echo "Started metro. PID=$metro_pid"
sleep 2s

# Start the test suite.
yarn test-integration
test_result=$?

# Kill the child process explicitly so that the exit code of the script
# isn't changed by it's eventual termination.
kill -TERM $metro_pid
sleep 2s
echo $metro_pid was terminated.

# Return the exit code of the test
exit $test_result
