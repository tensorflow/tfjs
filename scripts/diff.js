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

const {exec, constructDependencyGraph, calculateAffectedPackages} =
    require('./test-util');
const shell = require('shelljs');
const {readdirSync, statSync, writeFileSync} = require('fs');
const {join} = require('path');
const fs = require('fs');

const filesWhitelistToTriggerBuild = [
  'cloudbuild.yml', 'package.json', 'tsconfig.json', 'tslint.json',
  'scripts/run-build.sh'
];

const CLONE_PATH = 'clone';

const dirs = readdirSync('.').filter(f => {
  return f !== 'node_modules' && f !== '.git' && statSync(f).isDirectory();
});

let commitSha = process.env['COMMIT_SHA'];
let branchName = process.env['BRANCH_NAME'];
// If commit sha or branch name are null we are running this locally and are in
// a git repository.
if (commitSha == null) {
  commitSha = exec(`git rev-parse HEAD`).stdout.trim();
}
if (branchName == null) {
  branchName = exec(`git rev-parse --abbrev-ref HEAD`).stdout.trim();
}
console.log('commitSha: ', commitSha);
console.log('branchName: ', branchName);

// We cannot do --depth=1 here because we need to check out an old merge base.
// We cannot do --single-branch here because we need multiple branches.
exec(`git clone https://github.com/tensorflow/tfjs ${CLONE_PATH}`);

console.log();  // Break up the console for readability.

shell.cd(CLONE_PATH);

// If we cannot check out the commit then this PR is coming from a fork.
const res = shell.exec(`git checkout ${commitSha}`, {silent: true});
const isPullRequestFromFork = res.code !== 0;

// Only checkout the merge base if the pull requests comes from a
// tensorflow/tfjs branch. Otherwise clone master and diff against master.
if (!isPullRequestFromFork) {
  console.log('PR is coming from tensorflow/tfjs. Finding the merge base...');
  exec(`git checkout ${branchName}`);
  const mergeBase = exec(`git merge-base master ${branchName}`).stdout.trim();
  exec(`git fetch origin ${mergeBase}`);
  exec(`git checkout ${mergeBase}`);
  console.log('mergeBase: ', mergeBase);
} else {
  console.log('PR is coming from a fork. Diffing against master.');
}
shell.cd('..');
console.log();  // Break up the console for readability.

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

console.log();  // Break up the console for readability.

let triggeredBuilds = [];
dirs.forEach(dir => {
  shell.rm('-f', `${dir}/diff`);
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

console.log();  // Break up the console for readability.

if (!triggerAllBuilds) {
  let dependencyGraph = constructDependencyGraph('dependency.json');

  triggeredBuilds.forEach(triggeredBuild => {
    const affectedPackages =
        calculateAffectedPackages(dependencyGraph, triggeredBuild);
    if (affectedPackages.length > 0) {
      affectedPackages.forEach(package => {
        writeFileSync(join(package, 'diff'), '');
        triggeredBuild.push(package);
      })
    }
  });

  // Deduplicate the triggered builds for log.
  triggeredBuilds = [...new Set(triggeredBuilds)];
}

// Filter the triggered builds to log by whether a cloudbuild.yml file
// exists for that directory.
triggeredBuilds = triggeredBuilds.filter(
    triggeredBuild => fs.existsSync(triggeredBuild + '/cloudbuild.yml'));
console.log('Triggering builds for ', triggeredBuilds.join(', '));

function diff(fileOrDirName) {
  const diffCmd = `diff -rq --exclude='settings.json' ` +
      `${CLONE_PATH}/${fileOrDirName} ` +
      `${fileOrDirName}`;
  return exec(diffCmd, {silent: true}, true).stdout.trim();
}
