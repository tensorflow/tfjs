#!/usr/bin/env bash
# Copyright 2017 Google Inc. All Rights Reserved.
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

# Before you run this script, do this:
# 1) Update the version in package.json
# 2) Run ./scripts/build-npm from the base dir of the project.

# Then:
# 3) Checkout the master branch of this repo.
# 4) Run this script as `./scripts/publish-npm.sh` from the project base dir.

set -e

BRANCH=`git rev-parse --abbrev-ref HEAD`

if [ "$BRANCH" != "master" ]; then
  echo "Error: Switch to the master branch before publishing."
  exit
fi

yarn build-npm
npm publish
echo 'Yay! Published a new package to npm.'
