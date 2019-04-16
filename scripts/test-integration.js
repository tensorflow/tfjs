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

const {exec} = require('./test-util');

const dirName = 'tfjs-core-integration';

let shouldRunIntegration = false;
if (process.env.NIGHTLY === 'true') {
  shouldRunIntegration = true;
} else {
  exec(
      `git clone --depth=1 --single-branch ` +
      `https://github.com/tensorflow/tfjs-core.git ${dirName}`);
  const res = exec(
      `git diff --name-only --diff-filter=M --no-index ${dirName}/src/ src/`,
      {silent: true}, true);
  let files = res.stdout.trim().split('\n');
  files.forEach(file => {
    if (file === 'src/version.ts') {
      shouldRunIntegration = true;
    }
  });
}
if (shouldRunIntegration) {
  exec('./scripts/test-integration.sh');
}
