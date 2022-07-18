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
import * as shell from 'shelljs';
import {RELEASE_UNITS, question, $, printReleaseUnit, printPhase, getReleaseBranch, checkoutReleaseBranch, ALPHA_RELEASE_UNIT, TFJS_RELEASE_UNIT} from './release-util';
import * as fs from 'fs';

import {BAZEL_PACKAGES} from './bazel_packages';

const TMP_DIR = '/tmp/tfjs-publish';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

async function main() {
  const args = parser.parseArgs();

  RELEASE_UNITS.forEach(printReleaseUnit);
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

  const releaseUnit = RELEASE_UNITS[releaseUnitInt];
  const {name, phases} = releaseUnit;

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

  let releaseBranch: string;
  if (releaseUnit === ALPHA_RELEASE_UNIT) {
    // Alpha release unit is published with the tfjs release unit.
    releaseBranch = await getReleaseBranch(TFJS_RELEASE_UNIT.name);
  } else {
    releaseBranch = await getReleaseBranch(name);
  }
  console.log();

  checkoutReleaseBranch(releaseBranch, args.git_protocol, TMP_DIR);
  shell.cd(TMP_DIR);

  // Yarn in the top-level and in the directory.
  $('yarn');
  console.log();

  const packages = phases[phaseInt].packages;

  for (let i = 0; i < packages.length; i++) {
    const pkg = packages[i];
    shell.cd(pkg);

    // Check the package.json for 'link:' and 'file:' dependencies.
    const packageJson = JSON.parse(fs.readFileSync('package.json')
        .toString('utf8')) as {dependencies: Record<string, string>};
    if (packageJson.dependencies) {
      for (let [dep, depVersion] of Object.entries(packageJson.dependencies)) {
        const start = depVersion.slice(0,5);
        if (start === 'link:' || start === 'file:') {
          throw new Error(`${pkg} has a '${start}' dependency on ${dep}. `
                          + 'Refusing to publish.');
        }
      }
    }

    console.log(chalk.magenta.bold(`~~~ Preparing package ${pkg}~~~`));
    console.log(chalk.magenta('~~~ Installing packages ~~~'));
    // tfjs-node-gpu needs to get some files from tfjs-node.
    if (pkg === 'tfjs-node-gpu') {
      $('yarn prep-gpu');
    }

    // Yarn above the other checks to make sure yarn doesn't change the lock
    // file.
    $('yarn');

    console.log(chalk.magenta('~~~ Build npm ~~~'));

    if (pkg === 'tfjs-react-native' || BAZEL_PACKAGES.has(pkg)) {
      $('yarn build-npm');
    } else {
      $('yarn build-npm for-publish');
    }

    console.log(chalk.magenta.bold(`~~~ Publishing ${pkg} to npm ~~~`));

    const otp =
        await question(`Enter one-time password from your authenticator: `);

    if (BAZEL_PACKAGES.has(pkg)) {
      $(`YARN_REGISTRY="https://registry.npmjs.org/" yarn publish-npm -- -- --otp=${
          otp}`);
    } else {
      $(`YARN_REGISTRY="https://registry.npmjs.org/" npm publish --otp=${otp}`);
    }
    console.log(`Yay! Published ${pkg} to npm.`);

    shell.cd('..');
    console.log();
  }

  process.exit(0);
}

main();
