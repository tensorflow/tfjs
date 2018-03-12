#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

# Build pip package for keras_model_converter.

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# != 1 ]]; then
  echo "Usage:"
  echo "  build-pip-packages.sh <OUTPUT_DIR>"
  echo
  echo "Args:"
  echo "  OUTPUT_DIR: Directory where the pip (.whl) file will be written."
  exit 1
fi
DEST_DIR="$1"

mkdir -p "${DEST_DIR}"

yarn run prep

TMP_DIR=$(mktemp -d)

cp "${SCRIPTS_DIR}/../node_modules/deeplearn-src/scripts/write_weights.py" \
    "${TMP_DIR}/"
cp "${SCRIPTS_DIR}/h5_conversion.py" "${TMP_DIR}/"
cp "${SCRIPTS_DIR}/keras_model_converter.py" "${TMP_DIR}/"
cp "${SCRIPTS_DIR}/setup.py" "${TMP_DIR}/"

pushd "${TMP_DIR}" > /dev/null

python setup.py bdist_wheel
WHEELS=$(ls dist/*.whl)
mv dist/*.whl "${DEST_DIR}/"

popd > /dev/null

echo
echo "Generated wheel file(s) in ${DEST_DIR} :"
for WHEEL in ${WHEELS}; do
  echo "  $(basename "${WHEEL}")"
done

rm -rf "${TMP_DIR}"
