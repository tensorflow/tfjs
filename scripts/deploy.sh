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

# This script deploys demos to GCP so they can be statically hosted.
#
# The script can either be used without arguments, which deploys all demos:
# ./deploy.sh
# Or you can pass a single argument specifying a single demo to deploy:
# ./deploy.sh demos/mnist

cd demos
yarn

if [ -z "$1" ]
  then
    EXAMPLES=$(ls -d ./*/ | grep -v node_modules)
else
  EXAMPLES=$1
  if [ ! -d "$EXAMPLES" ]; then
    echo "Error: Could not find demo $1"
    echo "Make sure the first argument to this script matches the demo dir"
    exit 1
  fi
fi

prefix="./";
for i in $EXAMPLES; do
  npx rimraf dist/
  # Strip any trailing slashes and the ./ prefix

  EXAMPLE_NAME=${i%/}
  EXAMPLE_NAME=${EXAMPLE_NAME#$prefix};

  echo "building ${EXAMPLE_NAME}"

  yarn build-demo ${EXAMPLE_NAME}/index.html
  gsutil mkdir -p gs://tfjs-vis/$EXAMPLE_NAME
  gsutil mkdir -p gs://tfjs-vis/$EXAMPLE_NAME/dist
  gsutil -m cp dist/* gs://tfjs-vis/$EXAMPLE_NAME/dist

done

gsutil -m setmeta -h "Cache-Control:private" "gs://tfjs-vis/**.html"
gsutil -m setmeta -h "Cache-Control:private" "gs://tfjs-vis/**.css"
gsutil -m setmeta -h "Cache-Control:private" "gs://tfjs-vis/**.js"
