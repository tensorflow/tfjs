#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================


# Run this script from the base directory (not the script directory):
#   ./scripts/build-standalone.sh

set -e

IMPORT_NAME=tfjs_layers
PACKAGE_NAME=tfjs_layers
DIST_DIR=./dist
YARN=yarn
BROWSERIFY=node_modules/.bin/browserify
UGLIFYJS=node_modules/.bin/uglifyjs

"${YARN}" run prep
mkdir -p "${DIST_DIR}"
"${BROWSERIFY}" --standalone "${IMPORT_NAME}" src/index.ts -p [tsify] > "${DIST_DIR}/${PACKAGE_NAME}.js"
"${UGLIFYJS}" "${DIST_DIR}/${PACKAGE_NAME}.js" -c -m -o "${DIST_DIR}/${PACKAGE_NAME}.min.js"
"${BROWSERIFY}" --debug --standalone "${IMPORT_NAME}" src/index.ts -p [tsify] > "${DIST_DIR}/${PACKAGE_NAME}.debug.js"

echo "Stored standalone library at ${DIST_DIR}/${PACKAGE_NAME}(.min|.debug).js"
