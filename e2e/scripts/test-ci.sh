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

set -euxo pipefail

# Generate custom bundle files and model files for tests
./scripts/run-custom-builds.sh
./scripts/create-python-models.sh

TAGS='#SMOKE,#REGRESSION,#GOLDEN'

# Test with smoke/regression tests.
yarn karma start --single-run --tags "${TAGS}"

# Test script tag bundles
# Temporarily disabled
# yarn karma start --single-run ./script_tag_tests/tfjs/karma.conf.js --testBundle tf.min.js --tags "${TAGS}"
# yarn karma start --single-run ./script_tag_tests/tfjs-core-cpu/karma.conf.js --tags "${TAGS}"

# Test webpack
(cd webpack_test && yarn --mutex network && yarn build)
