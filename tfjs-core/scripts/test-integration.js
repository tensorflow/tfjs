#!/usr/bin/env node
// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const {exec} = require('../../scripts/test-util');
const fs = require('fs');


function shouldRunIntegration() {
  if (process.env.NIGHTLY === 'true') {
    return true;
  }
  const diffFile = 'diff';
  if (!fs.existsSync(diffFile)) {
    return false;
  }
  let diffContents = `${fs.readFileSync(diffFile)}`;
  if (diffContents.indexOf('src/version.ts') === -1) {
    return false;
  }
  return true;
}

if (shouldRunIntegration()) {
  exec('./scripts/test-integration.sh');
}
