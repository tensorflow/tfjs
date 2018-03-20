#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Run this script from the base directory (not the script directory):
# ./scripts/make-docs.sh

set -e

DIST_DIR=./dist
DOC_DIR="${DIST_DIR}/docs/api"
YARN=yarn
TYPEDOC=node_modules/.bin/typedoc

rm -rf "${DOC_DIR}"
"${YARN}"
"${TYPEDOC}" --out "${DOC_DIR}" --excludeExternals --excludeNotExported \
  --excludePrivate --mode file --tsconfig tsconfig-doc.json

echo "Created documentation in ${DOC_DIR}."
