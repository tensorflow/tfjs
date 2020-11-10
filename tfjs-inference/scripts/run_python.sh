#!/usr/bin/env bash

# Copyright 2020 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

cd python/

# Install python env.
if [[ -z "$(which pip3)" ]]; then
  echo "pip3 is not on path. Attempting to install it..."
  apt-get update
  apt-get install -y python3-pip python3-dev
fi

echo "Installing virtualenv..."
pip3 install virtualenv

VENV_DIR="$(mktemp -d)_venv"
echo "Creating virtualenv at ${VENV_DIR} ..."
virtualenv -p python3 "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install python packages.
pip3 install -r requirements.txt

# Run python tests.
python inference_test.py

# Clean up virtualenv directory.
rm -rf "${VENV_DIR}"

cd ..
