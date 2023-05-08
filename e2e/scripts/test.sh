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

# This script is used for local testing. Set environment variable TAGS to
# filter tests. If no TAGS is specified, the script will the TAGS to #SMOKE.
# Multiple tags are allowed, separate by comma.

set -e

if [[ -z "$TAGS" ]]; then
  echo "Env variable TAGS is not found, set TAGS='#SMOKE'"
  TAGS="#SMOKE"
fi

if [[ "$NIGHTLY" = true ]]; then
  TAGS="${TAGS},#GOLDEN,#REGRESSION"
fi

# Additional setup for regression tests.
if [[ "$TAGS" == *"#REGRESSION"* ]]; then
  # Generate canonical layers models and inputs.
  ./scripts/create-python-models.sh

  # Test webpack
  cd webpack_test
  yarn
  yarn build
  cd ..

  # Generate custom bundle files for tests
  ./scripts/run-custom-builds.sh
fi

echo "Karma tests."
yarn karma start --tags $TAGS
