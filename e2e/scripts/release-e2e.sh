#!/usr/bin/env bash
# Copyright 2020 Google LLC
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
# ==============================================================================

# This script is used for starting Verdaccio, a private npm registry.

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

function cleanup {
  echo 'Cleaning up.'
  # Restore the original NPM and Yarn registry URLs and stop Verdaccio
  stopLocalRegistry
}

# Error messages are redirected to stderr
function handle_error {
  echo "$(basename $0): ERROR! An error was encountered executing line $1." 1>&2;
  cleanup
  echo 'Exiting with error.' 1>&2;
  exit 1
}

function handle_exit {
  cleanup
  echo 'Exiting without error.' 1>&2;
  exit
}

# Exit the script with a helpful error message when any error is encountered
trap 'set +x; handle_error $LINENO $BASH_COMMAND' ERR

# Cleanup before exit on any termination signal
trap 'set +x; handle_exit' SIGQUIT SIGTERM SIGINT SIGKILL SIGHUP

# Echo every command being executed
set -x

# Go to e2e root
cd ..
e2e_root_path=$PWD

# ****************************************************************************
# First, publish the monorepo.
# ****************************************************************************
# Load functions for working with local NPM registry (Verdaccio)
source "$e2e_root_path"/scripts/local-registry.sh

# Start the local NPM registry
startLocalRegistry "$e2e_root_path"/scripts/verdaccio.yaml

# Publish the monorepo and update package.json tfjs dependency to the
# published version.
"$e2e_root_path"/scripts/publish-tfjs-ci.sh

# ****************************************************************************
# Second, install the packages from local registry.
# ****************************************************************************
# First make sure npm install succeeds.
npm install
rm -rf node_modules
# Then install dependencies via yarn.
yarn

# ****************************************************************************
# Third, run integration tests against locally published version.
# ****************************************************************************
yarn test-ci

# Cleanup
cleanup
