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

# The output dir is stored in the first parameter.
# It is passed from genrule.
OUTPUT_DIR="$1"

# The default version.
CURRENT_VERSION=0.0.6

# Get the version from the second parameter.
# Default to the value in CURRENT_VERSION.
VERSION="${2:-${CURRENT_VERSION}}"

# Make sure the version is provided.
if [[ -z ${VERSION} ]]; then
  echo "version (the only parameter) is required"
  exit 1
fi

# Download the zipped lib to the output dir.
cd "${OUTPUT_DIR}" && { curl -O https://storage.googleapis.com/tfweb/${VERSION}/tflite_web_api.zip; cd -; }

# Unzip and delete the zipped file.
unzip "${OUTPUT_DIR}/tflite_web_api.zip" -d "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/tflite_web_api.zip"

