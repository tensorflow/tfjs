#!/usr/bin/env bash
# Copyright 2021 Google LLC. All Rights Reserved.
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
# =============================================================================

# Halt if a single command errors
set -e

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Get the version from the first parameter.
VERSION="$1"

# Make sure the version is provided.
if [[ -z ${VERSION} ]]; then
  echo "version (the only parameter) is required"
  exit 1
fi

# Copy the artifacts from GCP to the deps/ dir.
mkdir -p ../deps
GCP_DIR="gs://tfweb/${VERSION}/dist"
gsutil -m cp "${GCP_DIR}/*" ../deps/
