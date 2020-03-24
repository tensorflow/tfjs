#!/usr/bin/env bash

# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

# Regular testing.
yarn
yarn build-deps-ci
yarn build-ci
yarn lint
yarn test-node
yarn test-browser-ci
