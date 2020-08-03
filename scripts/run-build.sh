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

# Echo every command being executed
set -x

# Exit the script on any command with non 0 return code
set -e

DIR=$1
# Regular flow: Only run changed packages plus e2e regular test.
# Nightly flow: Run everything.
if [[ -f "$DIR/run-ci" || $DIR == "e2e" || $NIGHTLY = true ]]; then
  gcloud builds submit . --config=$DIR/cloudbuild.yml \
    --substitutions _NIGHTLY=$NIGHTLY
fi
