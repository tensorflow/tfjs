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
if [ "$1" == "against-head" ]; then
  # Copy in fresh builds of all packages
  echo "Copying all deps from from HEAD"
  yarn copydeps
else
  # Copy just the built tfjs-react-native
  echo "Copying tfjs-react-native from HEAD"
  yarn && yarn copyrn
fi

yarn prep-tests

# Spawn a metro bundler/asset server in the background.
yarn start-metro &
let metro_pid=$!
echo "Started metro. PID=$metro_pid"
sleep 2s

# Start the test suite.
node ../../scripts/run_flaky.js "yarn test-integration"
test_result=$?

# Kill the child process explicitly so that the exit code of the script
# isn't changed by it's eventual termination.
kill -TERM $metro_pid
sleep 2s
echo $metro_pid was terminated.

# Return the exit code of the test
exit $test_result
