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
if [ -z "$1" ]
  then
    echo "No version number found."
    exit
fi

./scripts/build-standalone.sh $1 || exit 1

# Push to GCP.
echo "Pushing library to GCP..."

gsutil cp dist/deeplearn-$1.js gs://learnjs-data/
gsutil cp dist/deeplearn-$1.min.js gs://learnjs-data/
gsutil cp dist/deeplearn.js gs://learnjs-data/
gsutil cp dist/deeplearn.min.js gs://learnjs-data/
gsutil cp dist/deeplearn-latest.js gs://learnjs-data/
gsutil cp dist/deeplearn-latest.min.js gs://learnjs-data/

gsutil -m acl ch -u AllUsers:R -r gs://learnjs-data/*

echo "Stored standalone binaries in https://storage.googleapis.com/learnjs-data/."
