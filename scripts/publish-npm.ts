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

/*
 * This script publish to npm for all the TensorFlow.js packages. Before you run
 * this script, run `yarn release` and commit the PRs.
 * Then run this script as `yarn publish-npm`.
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as mkdirp from 'mkdirp';
import * as shell from 'shelljs';
import {RELEASE_UNITS, question, $, printReleaseUnit, printPhase} from './release-util';

const TMP_DIR = '/tmp/tfjs-publish';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

async function main() {
  const args = parser.parseArgs();

  RELEASE_UNITS.forEach((_, i) => printReleaseUnit(i));
  console.log();

  const releaseUnitStr =
      await question('Which release unit (leave empty for 0): ');
  const releaseUnitInt = +releaseUnitStr;
  if (releaseUnitInt < 0 || releaseUnitInt >= RELEASE_UNITS.length) {
    console.log(chalk.red(`Invalid release unit: ${releaseUnitStr}`));
    process.exit(1);
  }
  console.log(chalk.blue(`Using release unit ${releaseUnitInt}`));
  console.log();

  const {name, phases} = RELEASE_UNITS[releaseUnitInt];

  phases.forEach((_, i) => printPhase(phases, i));
  console.log();

  const phaseStr = await question('Which phase (leave empty for 0): ');
  const phaseInt = +phaseStr;
  if (phaseInt < 0 || phaseInt >= phases.length) {
    console.log(chalk.red(`Invalid phase: ${phaseStr}`));
    process.exit(1);
  }
  console.log(chalk.blue(`Using phase ${phaseInt}`));
  console.log();

  // Infer release branch name.
  let releaseBranch = '';

  // Get a list of branches sorted by timestamp in descending order.
  const branchesStr = $(
      `git branch -r --sort=-authordate --format='%(HEAD) %(refname:lstrip=-1)'`);
  const branches =
      Array.from(branchesStr.split(/\n/)).map(line => line.toString().trim());

  // Find the latest matching branch, e.g. tfjs_1.7.1
  // It will not match temprary generated branches such as tfjs_1.7.1_phase0.
  const exp = '^' + name + '_([^_]+)$';
  const regObj = new RegExp(exp);
  const maybeBranch = branches.find(branch => branch.match(regObj));
  releaseBranch = await question(`Which branch to publish from
  (leave empty for ${maybeBranch}): `);
  if (releaseBranch === '') {
    releaseBranch = maybeBranch;
  }
  console.log();

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
  // Yarn in the top-level and in the directory.
  $('yarn');
  console.log();

  const packages = phases[phaseInt].packages;

  for (let i = 0; i < packages.length; i++) {
    const pkg = packages[i];
    shell.cd(pkg);

    console.log(chalk.magenta.bold(`~~~ Preparing package ${pkg}~~~`));
    console.log(chalk.magenta('~~~ Installing packages ~~~'));
    // tfjs-node-gpu needs to get some files from tfjs-node.
    if (pkg === 'tfjs-node-gpu') {
      $('yarn prep-gpu');
    }

    // tfjs-backend-wasm needs emsdk to build.
    if (pkg === 'tfjs-backend-wasm') {
      shell.cd('..');
      $('git clone https://github.com/emscripten-core/emsdk.git');
      shell.cd('./emsdk');
      $('./emsdk install 1.39.15');
      $('./emsdk activate 1.39.15');
      shell.cd('..');
      shell.cd(pkg);
    }

    // Yarn above the other checks to make sure yarn doesn't change the lock
    // file.
    $('yarn');

    console.log(chalk.magenta('~~~ Build npm ~~~'));

    if (pkg === 'tfjs-backend-wasm') {
      // tfjs-backend-wasm needs emsdk env variables to build.
      $('source ../emsdk/emsdk_env.sh && yarn build-npm for-publish');
    } else if (pkg === 'tfjs-react-native') {
      $('yarn build-npm');
    } else {
      $('yarn build-npm for-publish');
    }

    console.log(chalk.magenta('~~~ Tag version ~~~'));
    shell.cd('..');
    const tagVersion = $(`./scripts/tag-version.js ${pkg}`);
    console.log(tagVersion);

    console.log(chalk.magenta.bold(`~~~ Publishing ${pkg} to npm ~~~`));
    shell.cd(pkg);
    const otp =
        await question(`Enter one-time password from your authenticator: `);
    $(`YARN_REGISTRY="https://registry.npmjs.org/" npm publish --otp=${otp}`);
    console.log(`Yay! Published ${pkg} to npm.`);

    shell.cd('..');
    console.log();
  }

  process.exit(0);
}

main();
