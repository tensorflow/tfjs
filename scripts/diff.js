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
const {readdirSync, statSync, writeFileSync} = require('fs');
const {join} = require('path');

const CLONE_PATH = 'clone';

const dirs = readdirSync('.').filter(f => {
  return f !== 'node_modules' && f !== '.git' && statSync(f).isDirectory();
});

exec(
    `git clone --depth=1 --single-branch ` +
    `https://github.com/tensorflow/tfjs-core.git ${CLONE_PATH}`);


dirs.forEach(dir => {
  const diffCmd = `diff -rq ${CLONE_PATH}/${dir}/ ./${dir}/`;
  const diffOutput = exec(diffCmd, {silent: true}, true).stdout.trim();

  if (diffOutput !== '') {
    console.log(`${dir} has modified files.`);
    writeFileSync(join(dir, 'diff'), diffOutput);
  } else {
    console.log(`No modified files found in ${dir}`);
  }
});
