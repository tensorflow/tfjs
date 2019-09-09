#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

yarn build-addon-from-source
yarn build
yarn lint
yarn test
