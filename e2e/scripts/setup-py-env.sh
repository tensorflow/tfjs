#!/usr/bin/env bash

# Copyright 2020 Google LLC
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

# Decide which environment to use.
DEV_VERSION=""
while [[ ! -z "$1" ]]; do
  if [[ "$1" == "--stable" ]]; then
    DEV_VERSION="stable"
  elif [[ "$1" == "--dev" ]]; then
    DEV_VERSION="dev"
  else
    echo "ERROR: Unrecognized command-line flag $1"
    exit 1
  fi
  shift
done

echo "DEV_VERSION: ${DEV_VERSION}"

if [[ -z "${DEV_VERSION}" ]]; then
  echo "Must specify one of --stable and --dev."
  exit 1
fi

VENV_DIR="$(mktemp -d)_venv"
echo "Creating virtualenv at ${VENV_DIR} ..."
PLATFORM="$(python -m platform)"
if [[ $PLATFORM =~ "Windows" ]]
then
  python -m virtualenv -p python3 "${VENV_DIR}"
  source "${VENV_DIR}/Scripts/activate"
else
  virtualenv -p python3 "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
fi

# Install python packages.
if [[ "${DEV_VERSION}" == "stable" ]]; then
  pip3 install -r requirements-stable.txt
else
  pip3 install -r requirements-dev.txt
fi

echo "Loading tensorflowjs pip from source ...."
pip3 install -e ../../tfjs-converter/python
