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

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Get release version from tfjs-core's package.json file.
function getReleaseVersion {
  local version=""
  local regex="\"version\": \"(.*)\""
  while read line
  do
    if [[ $line =~ $regex ]]; then
      version="${BASH_REMATCH[1]}"
      break
    fi
  done < "tfjs-core/package.json"
  echo "$version"
}

# Exit the script on any command with non 0 return code
set -e

# Echo every command being executed
set -x

# Go to root
cd ../../

# Yarn in the top-level
yarn

RELEASE_VERSION=`getReleaseVersion`

if [[ -z ${RELEASE_VERSION} ]]; then
  echo "Expect a valid release version, but got ${RELEASE_VERSION}"
  exit 1
else
  echo "Publishing version ${RELEASE_VERSION}"
fi

# Packages to publish.
PACKAGES=("tfjs-core" "tfjs-backend-cpu" "tfjs-backend-webgl" \
"tfjs-backend-wasm" "tfjs-layers" "tfjs-converter" "tfjs-data" "tfjs" \
"tfjs-node" "tfjs-node-gpu")

# Packages that build with Bazel
BAZEL_PACKAGES=("tfjs-core" "tfjs-backend-cpu" "tfjs-tflite")

for package in "${PACKAGES[@]}"
do
  cd $package

  # tfjs-node-gpu needs to get some files from tfjs-node.
  if [[ $package == "tfjs-node-gpu" ]]; then
    yarn prep-gpu
  fi

  # Install dependencies.
  yarn

  if [[ " ${BAZEL_PACKAGES[@]} " =~ " ${package} " ]]; then
    # Build and publish to local npm.
    yarn publish-npm
  else
    # Build npm.
    echo $package
    yarn build-npm for-publish

    # Publish to local npm.
    npm publish
  fi
  echo "Published ${package}@${RELEASE_VERSION}"

  cd ..
done

# Update e2e's package.json's all tfjs related packages to locally published
# version.
cd e2e
npx ts-node ./scripts/update-dependency.ts --version=$RELEASE_VERSION
