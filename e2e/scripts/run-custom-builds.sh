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

set -e

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"


# Go to e2e root
cd ..

echo "Building blazeface..."
(cd custom_module/blazeface && ./build.sh)

echo "Building dense_model..."
(cd custom_module/dense_model && ./build.sh)

echo "Building universal_sentence_encoder..."
(cd custom_module/universal_sentence_encoder && ./build.sh)
