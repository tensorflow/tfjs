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

set -e

# Smoke and regression tests run in PR and nightly builds.
TAGS="#SMOKE,#REGRESSION"

# Generate canonical layers models and inputs.
./scripts/create_save_predict.sh

cd integration_tests

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

# Generate custom bundle files for tests
./scripts/run-custom-builds.sh

# Test webpack
cd webpack_test
yarn
yarn build
cd ..

node ../scripts/run_flaky.js "yarn run-browserstack --browsers=win_10_chrome --tags '$TAGS'"

# Test script tag bundles
node ../scripts/run_flaky.js "karma start ./script_tag_tests/tfjs/karma.conf.js --browserstack --browsers=bs_chrome_mac --testBundle tf.min.js"

# Additional tests to run in nightly only.
if [[ "$NIGHTLY" = true || "$RELEASE" = true ]]; then
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_ios_12 --tags '$TAGS' --testEnv webgl --flags '{"\""WEBGL_VERSION"\"": 1, "\""WEBGL_CPU_FORWARD"\"": false, "\""WEBGL_SIZE_UPLOAD_UNIFORM"\"": 0}'"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_safari_mac --tags '$TAGS' --testEnv webgl --flags '{"\""WEBGL_VERSION"\"": 1, "\""WEBGL_CPU_FORWARD"\"": false, "\""WEBGL_SIZE_UPLOAD_UNIFORM"\"": 0}'"

  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_firefox_mac --tags '$TAGS'"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_chrome_mac --tags '$TAGS'"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_android_9 --tags '$TAGS'"

  # Test script tag bundles
  node ../scripts/run_flaky.js "karma start ./script_tag_tests/tfjs-core-cpu/karma.conf.js --browserstack --browsers=bs_chrome_mac"
fi
