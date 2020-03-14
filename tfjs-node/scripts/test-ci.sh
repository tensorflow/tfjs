#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

yarn build-addon-from-source
yarn build-core-ci
yarn build-layers-ci
yarn build-converter-ci
yarn build-data-ci
yarn build-union-ci
yarn build-ci
yarn lint
yarn test
