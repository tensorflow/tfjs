#!/usr/bin/env bash
# Copyright 2023 Google LLC
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

STORAGE_URL="https://storage.googleapis.com/tfjs-e2e-graph-model-golden-data"

cd ./integration_tests/graph_model_golden_data

golden_files=$(python -c "import json; print('\n'.join(json.load(open('./filenames.json'))))")

while read golden_file; do
  url="$STORAGE_URL/$golden_file"
  echo "Downloading $url"
  curl -O "$url"
done <<<"$(echo "$golden_files")"
