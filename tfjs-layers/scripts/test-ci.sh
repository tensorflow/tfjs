#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

# Regular testing.
node ../scripts/run_flaky.js "yarn run-browserstack"
