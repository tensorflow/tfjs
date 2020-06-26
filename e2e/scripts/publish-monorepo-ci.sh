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
root_path=$PWD

# Yarn in the top-level
yarn

# Todo(linazhao): publish monorepo
RELEASE_VERSION=`getReleaseVersion`

if [[ -z ${RELEASE_VERSION} ]]; then
  echo "Expect a valid release version, but got ${RELEASE_VERSION}"
  exit 1
else
  echo "Publishing version ${RELEASE_VERSION}"
fi

PACKAGES=("tfjs-core" "tfjs-backend-cpu" "tfjs-backend-webgl" \
"tfjs-backend-wasm" "tfjs-layers" "tfjs-converter" "tfjs-data" "tfjs" \
"tfjs-node" "tfjs-node-gpu")

for package in "${PACKAGES[@]}"
do
  cd $package

  # tfjs-node-gpu needs to get some files from tfjs-node.
  if [[ $package == "tfjs-node-gpu" ]]; then
    yarn prep-gpu
  fi

  # tfjs-backend-wasm needs emsdk to build.
  # if [[ $package == "tfjs-backend-wasm" ]]; then
  #   cd ..
  #   git clone https://github.com/emscripten-core/emsdk.git
  #   cd ./emsdk
  #   ./emsdk install 1.39.15
  #   ./emsdk activate 1.39.15
  #   source ./emsdk_env.sh
  #   cd ..
  #   cd $package
  # fi

  # Install dependencies.
  yarn

  # Build npm.
  yarn build-npm for-publish

  # Publish to local npm.
  npm publish
  echo "Published ${package}@${RELEASE_VERSION}"

  cd ..
done
