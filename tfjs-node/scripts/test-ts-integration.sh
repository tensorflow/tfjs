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

function print_status() {
  name=$1
  code=$2
  echo -e "$name:" $([[ $code -eq 0 ]] && echo 'PASS' || echo 'FAIL') $3
}

function test() {
  yarn && yarn build-addon-from-source && yarn build && yarn yalc publish

  cd integration/typescript
  yarn && yarn prep && yarn test
  TS_INTEGRATION_TEST_EXIT_CODE=$?
  git checkout package.json

  echo '==== TYPESCRIPT INTEGRATION TEST RESULTS ===='
  print_status "Exit code" "$TS_INTEGRATION_TEST_EXIT_CODE"
  echo '=================================='

  RED='\033[0;31m'
  BLUE='\e[34m'
  NC='\033[0m' # No Color
  FINAL_EXIT_CODE=$(($TS_INTEGRATION_TEST_EXIT_CODE))
  [[ $FINAL_EXIT_CODE -eq 0 ]] && COLOR=$BLUE || COLOR=$RED
  print_status "${COLOR}Final result" "$FINAL_EXIT_CODE" "$NC"
  exit $FINAL_EXIT_CODE
}

test
