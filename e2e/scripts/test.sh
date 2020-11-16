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
    TAGS="${TAGS},#REGRESSION"
fi

# Additional setup for regression tests.
if [[ "$TAGS" == *"#REGRESSION"*  ]]; then
  # Generate canonical layers models and inputs.
  ./scripts/create_save_predict.sh

  cd integration_tests

  # Setup python env.
  source ../scripts/setup-py-env.sh --dev

  echo "Load equivalent keras models and generate outputs."
  python create_save_predict.py

  echo "Create saved models and convert."
  python convert_predict.py

  echo "Convert model with user defined metadata."
  python metadata.py

  # Cleanup python env.
  source ../scripts/cleanup-py-env.sh

  cd ..


  # Build the wasm backend
  yarn build-backend-wasm

  # Generate custom bundle files for tests
  ./scripts/run-custom-builds.sh
fi

echo "Karma tests."
karma start --tags $TAGS

