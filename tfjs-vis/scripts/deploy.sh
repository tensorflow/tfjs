#!/usr/bin/env bash
# Copyright 2018 Google LLC. All Rights Reserved.
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

# This script deploys examples to GCP so they can be statically hosted.
#
# The script can either be used without arguments, which deploys all demos:
# ./deploy.sh
# Or you can pass a single argument specifying a single demo to deploy:
# ./deploy.sh mnist
#
# This script assumes that a directory in this repo corresponds to an example.
#
# Example directories should have:
#  - package.json
#  - `yarn build` script which generates a dist/ folder in the example directory.

if [ -z "$1" ]
  then
    EXAMPLES="api mnist mnist_internals"
else
  EXAMPLES=$1
  if [ ! -d "$EXAMPLES" ]; then
    echo "Error: Could not find example $1"
    echo "Make sure the first argument to this script matches the example dir"
    exit 1
  fi
fi

cd demos
for i in $EXAMPLES; do
  cd ${i}
  # Strip any trailing slashes.
  EXAMPLE_NAME=${i%/}

  echo "building ${EXAMPLE_NAME}..."
  yarn
  rm -rf dist .cache
  yarn build
  # Remove files in the example directory (but not sub-directories).
  gsutil -m rm gs://tfjs-vis/$EXAMPLE_NAME/dist/*
  # Gzip and copy all the dist files.
  # The trailing slash is important so we get $EXAMPLE_NAME/dist/.
  gsutil -m cp -Z -r dist gs://tfjs-vis/$EXAMPLE_NAME/
  cd ..
done
