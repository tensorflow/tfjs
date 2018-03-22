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

TMP_DIR=$(mktemp -d)
echo "Using temporary directory: ${TMP_DIR}"

pushd "${SCRIPTS_DIR}" > /dev/null
PY_FILES=$(find ./ -name '*.py' ! -name '*_test.py')

echo
for PY_FILE in ${PY_FILES}; do
  echo "Copying ${PY_FILE}"
  PY_DIR=$(dirname ${PY_FILE})
  mkdir -p "${TMP_DIR}/${PY_DIR}"
  cp "${PY_FILE}" "${TMP_DIR}/${PY_DIR}"
done

# Copy README.md.
echo "Copying README.md"
cp "${SCRIPTS_DIR}/README.md" "${TMP_DIR}/"

# Copy setup.cfg
echo "Copying setup.cfg"
cp "${SCRIPTS_DIR}/setup.cfg" "${TMP_DIR}/"

echo
popd

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
