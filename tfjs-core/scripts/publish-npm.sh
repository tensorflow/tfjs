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
# 2) Run ./scripts/make-version from the base dir of the project.
# 3) Run `yarn` to update `yarn.lock`, in case you updated dependencies
# 4) Commit to the master branch.

# Then:
# 5) Checkout the master branch of this repo.
# 6) Run this script as `./scripts/publish-npm.sh` from the project base dir.

set -e

BRANCH=`git rev-parse --abbrev-ref HEAD`
ORIGIN=`git config --get remote.origin.url`
CHANGES=`git status --porcelain`

if [ "$BRANCH" != "master" ]; then
  echo "Error: Switch to the master branch before publishing."
  exit
fi

if ! [[ "$ORIGIN" =~ tensorflow/tfjs-core ]]; then
  echo "Error: Switch to the main repo (tensorflow/tfjs-core) before publishing."
  exit
fi

if [ ! -z "$CHANGES" ];
then
    echo "Make sure the master branch is clean. Found changes:"
    echo $CHANGES
    exit 1
fi

yarn build-npm
./scripts/make-version # This is for safety in case you forgot to do 2).
./scripts/tag-version.js
npm publish
echo 'Yay! Published a new package to npm.'
