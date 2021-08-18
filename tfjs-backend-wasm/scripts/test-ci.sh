#!/usr/bin/env bash

# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

# Regular testing.
yarn test-node

if [ "$NIGHTLY" = true ]; then
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_safari_mac"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_firefox_mac"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_chrome_mac"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=win_10_chrome"
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_ios_11"
else
  node ../scripts/run_flaky.js "yarn run-browserstack --browsers=bs_chrome_mac"
fi
