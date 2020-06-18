#!/usr/bin/env bash
# Copyright 2020 Google LLC. All Rights Reserved.
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

# This script is used for starting Verdaccio, a private npm registry.

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Load functions for working with local NPM registry (Verdaccio)
source local-registry.sh

function cleanup {
  echo 'Cleaning up.'
  cd "$root_path"
  # Restore the original NPM and Yarn registry URLs and stop Verdaccio
  stopLocalRegistry
}

# Go to root
cd ..
root_path=$PWD

#if [[ "$RELEASE" = true ]]; then
  # ****************************************************************************
  # First, publish the monorepo.
  # ****************************************************************************

  # Start the local NPM registry
  startLocalRegistry "$root_path"/scripts/verdaccio.yaml

  # Publish the monorepo
  publishToLocalRegistry

  # Cleanup
  cleanup
#fi
