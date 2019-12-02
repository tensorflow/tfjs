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

# Utility script to update the integration-test app apk

set -e

echo "$(pwd)"
cd integration_rn59/android && ./gradlew clean assembleDebug && cd ../../
echo "$(pwd)"
gsutil cp ./integration_rn59/android/app/build/outputs/apk/debug/app-debug.apk gs://tfjs-rn/integration-tests/
gcloud pubsub topics publish sync_reactnative --message "{}"
