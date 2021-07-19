#!/usr/bin/env bash
# Copyright 2017 Google LLC. All Rights Reserved.
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

set -e

BRANCH=`git rev-parse --abbrev-ref HEAD`
ORIGIN=`git config --get remote.origin.url`
CHANGES=`git status --porcelain`

# Yarn in the top-level and in the directory,
yarn
cd $1
# Yarn above the other checks to make sure yarn doesn't change the lock file.
yarn
cd ..

PACKAGE_JSON_FILE="$1/package.json"
if ! test -f "$PACKAGE_JSON_FILE"; then
  echo "$PACKAGE_JSON_FILE does not exist."
  echo "Please pass the package name as the first argument."
  exit 1
fi

if [ "$BRANCH" != "master" ]; then
  echo "Error: Switch to the master branch before publishing."
  exit
fi

if ! [[ "$ORIGIN" =~ tensorflow/tfjs ]]; then
  echo "Error: Switch to the main repo (tensorflow/tfjs) before publishing."
  exit
fi

if [ ! -z "$CHANGES" ];
then
    echo "Make sure the master branch is clean. Found changes:"
    echo $CHANGES
    exit 1
fi

./scripts/make-version.js $1

cd $1
yarn build-npm for-publish
cd ..

./scripts/tag-version.js $1

cd $1
npm publish
echo 'Yay! Published a new package to npm.'
