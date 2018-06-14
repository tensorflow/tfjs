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

if [[ $(node -v) = *v10* ]]; then
  karma start \
      --browsers='bs_firefox_mac,bs_chrome_mac,bs_safari_mac,bs_ios_11' \
      --singleRun --reporters='dots,karma-typescript,BrowserStack' \
      --hostname='bs-local.com'
fi
