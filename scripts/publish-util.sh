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

# tfjs packages to publish, order matters, do not change.
PACKAGES=("tfjs-core" "tfjs-backend-cpu" "tfjs-backend-webgl" \
"tfjs-backend-wasm" "tfjs-layers" "tfjs-converter" "tfjs-data" "tfjs" \
"tfjs-node" "tfjs-node-gpu")

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

  if [[ -z ${version} ]]; then
    echo "Expect a valid release version, but got ${version}"
    exit 1
  else
    echo "$version"
  fi
}

function publishTfjs {
  echo "Publish tfjs@$1"

  # Go to root
  cd $root_path

  # Yarn in the top-level
  yarn

  for package in "${PACKAGES[@]}"
  do
    cd $package

    # tfjs-node-gpu needs to get some files from tfjs-node.
    if [[ $package == "tfjs-node-gpu" ]]; then
      yarn prep-gpu
    fi

    # Install dependencies.
    yarn

    # Build npm.
    yarn build-npm for-publish

    # Publish to local npm.
    npm publish
    echo "Published ${package}@$1"

    cd ..
  done
}
