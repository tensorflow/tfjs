#!/usr/bin/env bash
# Copyright 2023 Google LLC
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

# This script runs browserstack tests on all configured browsers. It requires
# the TAGS variable to be set in the environment.

set -e

# Smoke and regression tests run in PR and nightly builds.
TAGS="#SMOKE,#REGRESSION"
TAGS_WITH_GOLDEN="$TAGS,#GOLDEN"

# Test macOS with smoke/regression tests.
# Skip golden tests because they time out on browserstack (they work locally).
# TODO(mattSoulanille): Make golden tests work on BrowserStack Mac.
COMMANDS+=("yarn run-browserstack --browsers=bs_chrome_mac --tags '$TAGS'")

# Test windows 10 with smoke/regression/golden tests.
COMMANDS+=("yarn run-browserstack --browsers=win_10_chrome --tags '$TAGS_WITH_GOLDEN'")

# Test script tag bundles
COMMANDS+=("karma start ./script_tag_tests/tfjs/karma.conf.js --browserstack --browsers=bs_chrome_mac --testBundle tf.min.js")

# Additional tests to run in nightly only.
if [[ "$NIGHTLY" = true || "$RELEASE" = true ]]; then
  COMMANDS+=(
    "yarn run-browserstack --browsers=bs_ios_12 --tags '$TAGS' --testEnv webgl --flags '{\"\\"\"WEBGL_VERSION\"\\"\": 1, \"\\"\"WEBGL_CPU_FORWARD\"\\"\": false, \"\\"\"WEBGL_SIZE_UPLOAD_UNIFORM\"\\"\": 0}'"
    "yarn run-browserstack --browsers=bs_safari_mac --tags '$TAGS' --testEnv webgl --flags '{\"\\"\"WEBGL_VERSION\"\\"\": 1, \"\\"\"WEBGL_CPU_FORWARD\"\\"\": false, \"\\"\"WEBGL_SIZE_UPLOAD_UNIFORM\"\\"\": 0}'"
    "yarn run-browserstack --browsers=bs_firefox_mac --tags '$TAGS'"
    "yarn run-browserstack --browsers=bs_android_10 --tags '$TAGS'"
    # Test script tag bundles
    "karma start ./script_tag_tests/tfjs-core-cpu/karma.conf.js --browserstack --browsers=bs_chrome_mac"
  )
fi

for command in "${COMMANDS[@]}"; do
  TO_RUN+=("node ../scripts/run_flaky.js \"$command\"")
done

parallel ::: "${TO_RUN[@]}"
