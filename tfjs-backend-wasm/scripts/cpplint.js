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
const {exec} = require('../../scripts/test-util');
const fs = require('fs');

const CC_FILEPATH = 'src/cc';

let python2Cmd;

const ignoreCode = true;
const commandOpts = null;

let pythonVersion = exec('python --version', commandOpts, ignoreCode);
if (pythonVersion['stderr'].includes('Python 2')) {
  python2Cmd = 'python';
} else {
  pythonVersion = exec('python2 --version', commandOpts, ignoreCode);
  if (pythonVersion.code === 0) {
    python2Cmd = 'python2';
  }
}

if (python2Cmd != null) {
  const result = shell.find('src/cc').filter(
      fileName => fileName.endsWith('.cc') || fileName.endsWith('.h'));

  const cwd = process.cwd() + '/' + CC_FILEPATH;
  const filenameArgument = result.join(' ');

  exec(`${python2Cmd} tools/cpplint.py --root ${cwd} ${filenameArgument}`);
} else {
  console.warn(
      'No python2.x version found - please install python2. ' +
      'cpplint.py only works correctly with python 2.x.');
}
