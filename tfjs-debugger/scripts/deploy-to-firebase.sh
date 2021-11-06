#!/usr/bin/env bash
# Copyright 2021 Google LLC
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

# Deploy content in e2e/benchmarks to firebase hosting.

# Halt if a single command errors
set -e

PROJECT="jstensorflow"

if [[ "$1" == "--dev" ]]; then
  # Deploy to the preview channel which will expire in 14 days.
  #
  # Use the second command line parameter as the feature name which will be
  # part of the preview URL. Use the current date+time as fallback.
  feature_name="${2:-$(date '+%Y%m%d-%H%M%S')}"
  firebase hosting:channel:deploy \
    "${feature_name}" \
    --only tfjs-debugger \
    --project ${PROJECT} \
    --expires 14d
else
  firebase deploy \
    --only hosting:tfjs-debugger \
    --project ${PROJECT}
fi
