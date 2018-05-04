#!/usr/bin/env bash
# Copyright 2018 Google LLC
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
# ==============================================================================

# Build pip package for keras_model_converter.
#
# Run this script outside a virtualenv, as this script will activate virtualenvs
# for python2 and python3 to generate the wheel files.
#
# Usage:
#   build-pip-package.sh \
#       [--test] [--upload] [--upload-to-test] [--confirm-upload] <DEST_DIR>
#
# Positional arguments:
#   DEST_DIR: Destination directory for writing the pip wheels.
#
# Optional argumnets:
#   --test: Test the pip packages by installing it (inside virtualenv)
#           and running test_pip_package.py against the install.
#   --upload:         Upload the py2 and py3 wheels to prod PyPI.
#   --upload-to-test: Upload the py2 and py3 wheels to test PyPI, mutually
#                     exclusive with --upload.
#   --confirm-upload: Do not prompt for yes/no before uploading to test or
#                     prod PyPI. Use with care!
#
# N.B. For pypi/twine authentication, you need to have the file ~/.pypirc,
#   with content formatted like below:
# ```
# [distutils]
# index-servers =
#     pypi-warehouse
#     test-warehouse
#
# [test-warehouse]
#     repository = https://test.pypi.org/legacy/
#     username:<PYPIUSERNAME>
#     password:<PYPIPWD>
#
# [pypi-warehouse]
#     repository = https://upload.pypi.org/legacy/
#     username:<PYPIUSERNAME>
#     password:<PYPIPWD>
# ```

set -e

function print_usage() {
  echo "Usage:"
  echo "  build-pip-packages.sh \\"
  echo "      [--test] [--upload] [--upload-to-test] [--confirm-upload] <OUTPUT_DIR>"
  echo
}

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# == 0 ]]; then
  print_usage
  exit 1
fi

RUN_TEST=0
UPLOAD_TO_PROD_PYPI=0
UPLOAD_TO_TEST_PYPI=0
CONFIRM_UPLOAD=0
DEST_DIR=""
while true; do
  if [[ "$1" == "--test" ]]; then
    RUN_TEST=1
  elif [[ "$1" == "--upload" ]]; then
    UPLOAD_TO_PROD_PYPI=1
  elif [[ "$1" == "--upload-to-test" ]]; then
    UPLOAD_TO_TEST_PYPI=1
  elif [[ "$1" == "--confirm-upload" ]]; then
    CONFIRM_UPLOAD=1
  elif [[ "$1" != --* ]]; then
    DEST_DIR="$1"
  else
    print_usage
    exit 1
  fi
  shift

  if [[ -z "$1" ]]; then
    break
  fi
done

if [[ "${UPLOAD_TO_TEST_PYPI}" == 1 && "${UPLOAD_TO_PROD_PYPI}" == 1 ]]; then
  echo "ERROR: Do not use --upload and --upload-to-test together."
  exit 1
fi

if [[ -z "${DEST_DIR}" ]]; then
  print_usage
  exit 1
fi

if [[ -f "${DEST_DIR}" || -d "${DEST_DIR}" ]]; then
  echo "ERROR: ${DEST_DIR} already exists. Please specify a new DEST_DIR."
  exit 1
fi

mkdir -p "${DEST_DIR}"
DEST_DIR="$(cd "${DEST_DIR}" 2>/dev/null && pwd -P)"

# Copy all non-test .py files.
TMP_DIR="$(mktemp -d)"
echo "Using temporary directory: ${TMP_DIR}"

PY_FILES=$(find . -name '*.py' ! -name '*_test.py')
for PY_FILE in ${PY_FILES}; do
  echo "Copying ${PY_FILE}"
  PY_DIR=$(dirname "${PY_FILE}")
  mkdir -p "${TMP_DIR}/${PY_DIR}"
  cp "${PY_FILE}" "${TMP_DIR}/${PY_DIR}"
done

# Copy .json files under op_list
OP_LIST_DIR="tensorflowjs/op_list"
JSON_FILES=$(find -L "${SCRIPTS_DIR}/${OP_LIST_DIR}" -name '*.json')
if [[ -z "${JSON_FILES}" ]]; then
  echo "ERROR: Failed to find any .json files in ${SCRIPTS_DIR}/${OP_LIST_DIR}"
  exit 1
fi

mkdir -p "${TMP_DIR}/${OP_LIST_DIR}"
echo
for JSON_FILE in ${JSON_FILES}; do
  echo "Copying JSON file: $(basename "${JSON_FILE}")"
  cp "${JSON_FILE}" "${TMP_DIR}/${OP_LIST_DIR}"
done

# Copy requirements.txt
echo
echo "Copying requirements.txt"
cp "${SCRIPTS_DIR}/requirements.txt" "${TMP_DIR}/"

# Copy README.md.
echo
echo "Copying README.md"
cp "${SCRIPTS_DIR}/README.md" "${TMP_DIR}/"

# Copy setup.cfg
echo
echo "Copying setup.cfg"
cp "${SCRIPTS_DIR}/setup.cfg" "${TMP_DIR}/"

echo

# Check virtualenv is on path.
if [[ -z "$(which virtualenv)" ]]; then
  echo "ERROR: Cannot find virtualenv on path. Install virtualenv first."
  exit 1
fi

# Create virtualenvs for python2 and python3; build (and test) the wheels inside
# them.
VENV_PYTHON_BINS="python2 python3"
for VENV_PYTHON_BIN in ${VENV_PYTHON_BINS}; do
  if [[ -z "$(which "${VENV_PYTHON_BIN}")" ]]; then
    echo "ERROR: Unable to find ${VENV_PYTHON_BIN} on path."
    exit 1
  fi

  TMP_VENV_DIR="$(mktemp -d)"
  virtualenv -p "${VENV_PYTHON_BIN}" "${TMP_VENV_DIR}"
  source "${TMP_VENV_DIR}/bin/activate"

  echo
  echo "Building wheel for ${VENV_PYTHON_BIN}: $(python --version 2>&1) ..."
  echo

  pip install -r "${SCRIPTS_DIR}/requirements.txt"

  pushd "${TMP_DIR}" > /dev/null
  echo

  echo "Building wheel for $(python --version 2>&1) ..."
  echo

  python setup.py bdist_wheel
  WHEELS=$(ls dist/*.whl)
  mv dist/*.whl "${DEST_DIR}/"

  WHEEL_PATH=""
  echo
  echo "Generated wheel file(s) in ${DEST_DIR} :"
  for WHEEL in ${WHEELS}; do
    WHEEL_BASE_NAME="$(basename "${WHEEL}")"
    echo "  ${WHEEL_BASE_NAME}"
    WHEEL_PATH="${DEST_DIR}/${WHEEL_BASE_NAME}"
  done

  # Run test on install.
  if [[ "${RUN_TEST}" == "1" ]]; then
    echo
    echo "Running test-on-install for $(python --version 2>&1) ..."
    echo

    pip uninstall -y tensorflowjs || \
      echo "It appears that tensorflowjs is not installed."

    echo
    echo "Installing tensorflowjs from wheel at path: ${WHEEL_PATH} ..."
    echo

    TEST_ON_INSTALL_DIR="$(mktemp -d)"

    cp "${SCRIPTS_DIR}/test_pip_package.py" "${TEST_ON_INSTALL_DIR}"

    pushd "${TEST_ON_INSTALL_DIR}" > /dev/null

    pip install "${WHEEL_PATH}"
    echo "Successfully installed ${WHEEL_PATH} for $(python --version 2>&1)."
    echo

    python test_pip_package.py

    popd > /dev/null

    rm -rf "${TEST_ON_INSTALL_DIR}"

    echo
    echo "Test-on-install for $(python --version 2>&1) PASSED."
    echo
    echo "Your pip wheel for $(python --version 2>&1) is at:"
    echo "  ${WHEEL_PATH}"
  fi

  popd > /dev/null

  deactivate
  rm -rf "${TMP_VENV_DIR}"
done

if [[ "${UPLOAD_TO_PROD_PYPI}" == 1 || "${UPLOAD_TO_TEST_PYPI}" == 1 ]]; then
  if [[ "${UPLOAD_TO_PROD_PYPI}" == 1 ]]; then
    UPLOAD_DEST="pypi-warehouse"
  else
    UPLOAD_DEST="test-warehouse"
  fi

  pushd "${DEST_DIR}" > /dev/null
  echo "Found wheel files to upload to ${UPLOAD_DEST}:"
  ls ./*.whl
  echo

  TO_UPLOAD=0
  if [[ "${CONFIRM_UPLOAD}" == 0 ]]; then
    while true; do
      read -r -p "Do you wish to proceed with the upload to ${UPLOAD_DEST}? (y/n): " yn
        case $yn in
          [Yy]* ) TO_UPLOAD=1; break;;
          [Nn]* ) exit;;
          * ) echo "Please answer y or n.";;
        esac
    done
  else
    TO_UPLOAD=1
  fi

  if [[ "${TO_UPLOAD}" == 1 ]]; then
    echo "Proceeding with the upload ..."
    echo

    # Create a virtualenv and install twine in it for uploading.
    TMP_VENV_DIR="$(mktemp -d)"
    virtualenv -p "${VENV_PYTHON_BIN}" "${TMP_VENV_DIR}"
    source "${TMP_VENV_DIR}/bin/activate"
    pip install twine

    twine upload -r "${UPLOAD_DEST}" ./*.whl

    deactivate
    rm -rf "${TMP_VENV_DIR}"
  fi

  popd > /dev/null
fi

rm -rf "${TMP_DIR}"
