#!/usr/bin/env bash

# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

set -e

cd integration_tests/tfjs2keras
DEV_VERSION=""
TFJS2KERAS_TEST_USING_TF_KERAS=0
while [[ ! -z "$1" ]]; do
  if [[ "$1" == "--stable" ]]; then
    DEV_VERSION="stable"
  elif [[ "$1" == "--dev" ]]; then
    DEV_VERSION="dev"
  elif [[ "$1" == "--tfkeras" ]]; then
    TFJS2KERAS_TEST_USING_TF_KERAS=1
  else
    echo "ERROR: Unrecognized command-line flag $1"
    exit 1
  fi
  shift
done

echo "DEV_VERSION: ${DEV_VERSION}"
echo "TFJS2KERAS_TEST_USING_TF_KERAS: ${TFJS2KERAS_TEST_USING_TF_KERAS}"

if [[ -z "${DEV_VERSION}" ]]; then
  echo "Must specify one of --stable and --dev."
  exit 1
fi

if [[ "${DEV_VERSION}" == "dev" &&
      "${TFJS2KERAS_TEST_USING_TF_KERAS}" == "0" ]]; then
  echo "--dev && keras-team/keras is not a valid combination."
  echo "Use --dev and --tfkeras together."
  exit 1
fi

VENV_DIR="$(mktemp -d)_venv"
echo "Creating virtualenv at ${VENV_DIR} ..."
virtualenv -p python3 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

if [[ "${DEV_VERSION}" == "stable" ]]; then
  pip3 install -r requirements-stable.txt
else
  pip3 install -r requirements-dev.txt
fi

export TFJS2KERAS_TEST_USING_TF_KERAS="${TFJS2KERAS_TEST_USING_TF_KERAS}"

python tfjs2keras_test.py

# Clean up virtualenv directory.
rm -rf "${VENV_DIR}"
cd ../..
