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

# Start in scripts/ even if run from root directory.
cd "$(dirname "$0")"

# Exit the script on any command with non 0 return code.
set -e

# Echo every command being executed.
set -x

# master branch is our main dev branch, must not publish from master.
if [[ $BRANCH_NAME == 'master' ]]; then
  echo "Cannot publish from branch master."
  exit 1
fi

# Go to root
cd ../
root_path=$PWD

# Load functions for publish
source "$root_path"/scripts/publish-util.sh

RELEASE_VERSION=`getReleaseVersion`

publishTfjs $RELEASE_VERSION

echo "Publish is done. Remember to tag the release."
