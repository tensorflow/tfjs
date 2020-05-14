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

set -e

if [ "$NIGHTLY" = true ]; then
  yarn run-browserstack --browsers=bs_safari_mac,bs_ios_11 --testEnv webgl1 --flags '{"WEBGL_CPU_FORWARD": false, "WEBGL_SIZE_UPLOAD_UNIFORM": 0}'
  yarn run-browserstack --browsers=bs_firefox_mac,bs_chrome_mac
  yarn run-browserstack --browsers=bs_chrome_mac,win_10_chrome,bs_android_9 --testEnv webgl2 --flags '{"WEBGL_CPU_FORWARD": false, "WEBGL_SIZE_UPLOAD_UNIFORM": 0}'
  yarn run-browserstack --browsers=bs_chrome_mac --testEnv webgl2 --flags '{"WEBGL_PACK": false}'
else
  yarn run-browserstack --browsers=bs_chrome_mac
fi
