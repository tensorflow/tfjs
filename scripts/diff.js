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
const fs = require('fs');

const filesWhitelistToTriggerBuild = [
  'cloudbuild.yml', 'package.json', 'tsconfig.json', 'tslint.json',
  'scripts/diff.js', 'scripts/run-build.sh'
];

const CLONE_PATH = 'clone';

const dirs = readdirSync('.').filter(f => {
  return f !== 'node_modules' && f !== '.git' && statSync(f).isDirectory();
});

exec(
    `git clone --depth=1 --single-branch ` +
    `https://github.com/tensorflow/tfjs ${CLONE_PATH}`);

let triggerAllBuilds = false;
let whitelistDiffOutput = [];
filesWhitelistToTriggerBuild.forEach(fileToTriggerBuild => {
  const diffOutput = diff(fileToTriggerBuild);
  if (diffOutput !== '') {
    console.log(fileToTriggerBuild, 'has changed. Triggering all builds.');
    triggerAllBuilds = true;
    whitelistDiffOutput.push(diffOutput);
  }
});

// Break up the console for readability.
console.log();

let triggeredBuilds = [];
dirs.forEach(dir => {
  const diffOutput = diff(`${dir}/`);
  if (diffOutput !== '') {
    console.log(`${dir} has modified files.`);
  } else {
    console.log(`No modified files found in ${dir}`);
  }

  const shouldDiff = diffOutput !== '' || triggerAllBuilds;
  if (shouldDiff) {
    const diffContents = whitelistDiffOutput.join('\n') + '\n' + diffOutput;
    writeFileSync(join(dir, 'diff'), diffContents);
    triggeredBuilds.push(dir);
  }
});

// Break up the console for readability.
console.log();

// Filter the triggered builds to log by whether a cloudbuild.yml file exists
// for that directory.
triggeredBuilds = triggeredBuilds.filter(
    triggeredBuild => fs.existsSync(triggeredBuild + '/cloudbuild.yml'));
console.log('Triggering builds for ', triggeredBuilds.join(', '));

function diff(fileOrDirName) {
  const diffCmd = `diff -rq ${CLONE_PATH}/${fileOrDirName} ./${fileOrDirName}`;
  return exec(diffCmd, {silent: true}, true).stdout.trim();
}
