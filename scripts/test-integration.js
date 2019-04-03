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

const shell = require('shelljs');

function exec(command, opt, ignoreCode) {
  const res = shell.exec(command, opt);
  if (!ignoreCode && res.code !== 0) {
    shell.echo('command', command, 'returned code', res.code);
    shell.exit(1);
  }
  return res;
}

exec(
    `git clone --depth=1 --single-branch ` +
    `https://github.com/tensorflow/tfjs-core.git`);
const res = exec(
    'git diff --name-only --diff-filter=M --no-index tfjs-core/src/ src/',
    {silent: true}, true);
let files = res.stdout.trim().split('\n');
files.forEach(file => {
  if (file === 'src/version.ts') {
    shell.exec('./scripts/test-integration.sh');
  }
});
