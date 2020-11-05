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
const shell = require('shelljs');
const {readdirSync, statSync} = require('fs');
const {join} = require('path');


const filesAllowlistToTriggerBuild = [
  'cloudbuild.yml', 'package.json', 'tsconfig.json', 'tslint.json',
  'scripts/find_packages_with_diff.js', 'scripts/run-build.sh',
  'scripts/generate_cloudbuild.js'
];

const CLONE_PATH = 'clone';
let commitSha = process.env['COMMIT_SHA'];
let branchName = process.env['BRANCH_NAME'];
let baseBranch = process.env['BASE_BRANCH'];


const allPackages = readdirSync('.').filter(f => {
  if (f === 'node_modules' || f === '.git' || f === 'clone' ||
      !statSync(f).isDirectory()) {
    return false;
  }
  const directoryContents = readdirSync(join('.', f));
  return directoryContents.includes('cloudbuild.yml');
});


function findPackagesWithDiff() {
  // For Nightly build, baseBranch is one of the falsey values. We use master
  // for Nightly build.
  if (!baseBranch) {
    baseBranch = 'master';
  }
  console.log('commitSha: ', commitSha);
  console.log('branchName: ', branchName);
  console.log('baseBranch: ', baseBranch);

  // We cannot do --depth=1 here because we need to check out an old merge base.
  // We cannot do --single-branch here because we need multiple branches.
  console.log(`Clone branch ${baseBranch}`);
  shell.rm('-rf', CLONE_PATH);
  exec(`git clone -b ${baseBranch} https://github.com/tensorflow/tfjs ${
      CLONE_PATH}`);

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
    const mergeBase =
        exec(`git merge-base ${baseBranch} ${branchName}`).stdout.trim();
    exec(`git fetch origin ${mergeBase}`);
    exec(`git checkout ${mergeBase}`);
    console.log('mergeBase: ', mergeBase);
  } else {
    console.log(`PR is going to diff against branch ${baseBranch}.`);
  }
  shell.cd('..');
  console.log();  // Break up the console for readability.

  let triggerAllBuilds = false;
  let allowlistDiffOutput = [];
  filesAllowlistToTriggerBuild.forEach(fileToTriggerBuild => {
    const diffOutput = diff(fileToTriggerBuild);
    if (diffOutput !== '') {
      console.log(fileToTriggerBuild, 'has changed. Triggering all builds.');
      triggerAllBuilds = true;
      allowlistDiffOutput.push(diffOutput);
    }
  });

  console.log();  // Break up the console for readability.

  let packagesWithDiff = [];
  allPackages.forEach(dir => {
    const diffOutput = diff(`${dir}/`);
    if (diffOutput !== '') {
      console.log(`${dir} has modified files.`);
    } else {
      console.log(`No modified files found in ${dir}`);
    }

    const shouldDiff = diffOutput !== '' || triggerAllBuilds;
    if (shouldDiff) {
      packagesWithDiff.push(dir);
    }
  });

  console.log();  // Break up the console for readability.

  console.log(`Packages directly affected: ${packagesWithDiff.join(', ')}`);
  return packagesWithDiff;
}

function diff(fileOrDirName) {
  const diffCmd = `diff -rq --exclude='settings.json' ` +
      `${CLONE_PATH}/${fileOrDirName} ` +
      `${fileOrDirName}`;
  return exec(diffCmd, {silent: true}, true).stdout.trim();
}

exports.findPackagesWithDiff = findPackagesWithDiff;
exports.allPackages = allPackages;
