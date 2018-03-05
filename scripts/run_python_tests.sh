#!/usr/bin/env bash
# Copyright 2018 Google Inc. All Rights Reserved.
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

echo
echo "Installing python3 dependencies..."
pip install -r ./scripts/requirements.txt --user --quiet
echo "Running Python 2 unit tests..."
python -m unittest discover scripts "*_test.py"

echo
echo "Installing python3 dependencies..."
python3 -m pip install -r ./scripts/requirements.txt --user --quiet
echo "Running python3 unit tests..."
python3 -m unittest discover scripts "*_test.py"
