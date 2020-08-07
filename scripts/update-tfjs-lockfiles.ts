#!/usr/bin/env node
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * This script update yarn.lock after tfjs packages are published.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as shell from 'shelljs';

import {$, createPR, question, TFJS_RELEASE_UNIT, prepareReleaseBuild} from './release-util';

import mkdirp = require('mkdirp');

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
});

const TMP_DIR = '/tmp/tfjs-publish-after';

async function main() {
  const args = parser.parseArgs();

  // ========== Get release branch. ============================================
  // Infer release branch name.
  let releaseBranch = '';

  // Get a list of branches sorted by timestamp in descending order.
  const branchesStr = $(
      `git branch -r --sort=-authordate --format='%(HEAD) %(refname:lstrip=-1)'`);
  const branches =
      Array.from(branchesStr.split(/\n/)).map(line => line.toString().trim());

  // Find the latest matching branch, e.g. tfjs_1.7.1
  // It will not match temprary generated branches such as tfjs_1.7.1_phase0.
  const exp = '^tfjs_([^_]+)$';
  const regObj = new RegExp(exp);
  const maybeBranch = branches.find(branch => branch.match(regObj));
  releaseBranch = await question(`Which branch to update lockfiles for
  (leave empty for ${maybeBranch}): `);
  if (releaseBranch === '') {
    releaseBranch = maybeBranch;
  }
  console.log();


  // ========== Checkout release branch. =======================================
  console.log(chalk.magenta.bold(
      `~~~ Checking out release branch ${releaseBranch} ~~~`));
  $(`rm -f -r ${TMP_DIR}`);
  mkdirp(TMP_DIR, err => {
    if (err) {
      console.log('Error creating temp dir', TMP_DIR);
      process.exit(1);
    }
  });

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${
      TMP_DIR} --depth=1`);
  shell.cd(TMP_DIR);

  // ========== Run yarn to update yarn.lock file for each package. ============
  // Yarn in the top-level.
  $('yarn');

  // run yarn for every tfjs package
  const phases = TFJS_RELEASE_UNIT.phases;

  for (let i = 0; i < phases.length; i++) {
    const phase = phases[i];
    const packages = phase.packages;

    for (let i = 0; i < packages.length; i++) {
      const packageName = packages[i];
      shell.cd(packageName);

      prepareReleaseBuild(phase, packageName);

      shell.cd('..');
    }
  }

  // Use dev prefix to avoid branch being locked.
  const devBranchName = `dev_${releaseBranch}_update`;

  const message = `Update release branch ${releaseBranch} lock files.`;
  createPR(devBranchName, releaseBranch, message);

  console.log('Done.');

  process.exit(0);
}

main();
