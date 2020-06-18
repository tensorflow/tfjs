#!/usr/bin/env bash
# Copyright 2020 Google LLC. All Rights Reserved.
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

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Exit the script on any command with non 0 return code
set -e

# Echo every command being executed
set -x

# Go to root
cd ..
root_path=$PWD

if [ -n "$(git status --porcelain)" ]; then
  echo "Your git status is not clean. Aborting.";
  exit 1;
fi

# Publish to local registry
# Todo(linazhao): Develop and use a one step for all release script.
echo "publish to verdaccio"
