#!/usr/bin/env bash
# Copyright 2020 Google Inc. All Rights Reserved.
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

# Before you run this script, run `yarn release` and commit the PRs.

# Then:
# 1) Checkout the master branch of this repo.
# 2) Run this script as `./scripts/publish-npm.sh DIR_NAME`
#      from the project base dir where DIR_NAME is the directory name of the
#      package you want to publish, e.g. "tfjs-core".

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

set -e

# Exit the script on any command with non 0 return code
set -e

# Echo every command being executed
set -x

# Go to root
cd ..
root_path=$PWD

# Yarn in the top-level and in the directory,
yarn

# Publish tfjs-core
cd tfjs-core

# Yarn above the other checks to make sure yarn doesn't change the lock file.
yarn

if [ -n "$(git status --porcelain)" ]; then
  echo "Your git status is not clean. Aborting.";
  exit 1;
fi

yarn build-npm for-publish

npm publish
echo 'Yay! Published tfjs-core to npm.'
