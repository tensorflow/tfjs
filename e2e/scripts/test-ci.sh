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

set -e

# Smoke tests run in PR and nightly builds.
TAGS="#SMOKE"

# Regression tests run in nightly builds.
if [[ "$NIGHTLY" = true ]]; then
    TAGS="${TAGS},#REGRESSION"
fi

# Additional setup for regression tests.
if [[ "$TAGS" == *"#REGRESSION"*  ]]; then
    # Generate canonical layers models and inputs.
    ./scripts/tfjs2keras-js.sh
    # Load equivalent keras models and generate outputs.
    # TODO(linazhao): Investigate why --dev --tfkeras fail.
    ./scripts/tfjs2keras-py.sh --stable
fi

if [ "$NIGHTLY" = true ]; then
  yarn run-browserstack --browsers=bs_chrome_mac --tags $TAGS
  yarn run-browserstack --browsers=bs_safari_mac,bs_firefox_mac,win_10_chrome,bs_ios_11,bs_android_9 --tags $TAGS
else
  yarn run-browserstack --browsers=bs_chrome_mac --tags $TAGS
fi
