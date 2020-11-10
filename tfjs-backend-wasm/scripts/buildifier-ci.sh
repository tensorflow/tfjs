#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
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

yarn bazel:format-check

if [ $? -eq 0 ]
then
  echo
else
  echo
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "'yarn bazel:format-check' failed!"
  echo "Please run 'yarn bazel:format'."
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo
  exit 1
fi

yarn bazel:lint-check

if [ $? -eq 0 ]
then
  echo
else
  echo
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "'yarn bazel:lint-check' failed!"
  echo "Please run 'yarn bazel:lint'."
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo
  exit 1
fi
