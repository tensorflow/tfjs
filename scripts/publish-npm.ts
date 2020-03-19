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
import * as readline from 'readline';
import * as shell from 'shelljs';
import {RELEASE_UNITS, question, $, printReleaseUnit} from './release-util';
import {arrayify} from 'tslint/lib/utils';

const TMP_DIR = '/tmp/tfjs-publish';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

function printPackage(pkg: string, id: number) {
  console.log(chalk.green(`Package ${id}: `));
  console.log(chalk.blue(pkg));
}

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

  const {name, phases, repo} = RELEASE_UNITS[releaseUnitInt];

  const packages = phases.map(phase =>
    phase.packages).reduce((arr, packages) =>
      arr.concat(packages), []);
  packages.forEach((pkg, i) => printPackage(pkg, i));
  console.log();

  const pkgStr = await question('Which package to publish (leave empty for 0): ');
  const pkgInt = +pkgStr;
  if (pkgInt < 0 || pkgInt >= packages.length) {
    console.log(chalk.red(`Invalid package: ${pkgStr}`));
    process.exit(1);
  }
  const pkg = packages[pkgInt];
  console.log(chalk.blue(`Publishing package ${pkg}`));
  console.log();

  // Infer release branch name, does not apply to package0.
  let releaseBranch = '';
  const latestVersion = pkgInt !== 0 ?
    $(`npm view @tensorflow/${pkg} dist-tags.latest`) :
    await question('What is the release version: ');
  releaseBranch = `${name}_${latestVersion}`;
  console.log();

  console.log(`~~~ Checking out release branch ${releaseBranch} ~~~`);
  $(`rm -f -r ${TMP_DIR}`);
  mkdirp(TMP_DIR, err => {
    if (err) {
      console.log('Error creating temp dir', TMP_DIR);
      process.exit(1);
    }
  });

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${TMP_DIR} --depth=1`);
  shell.cd(TMP_DIR);
  console.log();

  console.log('~~~ Installing packages ~~~');
  // Yarn in the top-level and in the directory.
  $('yarn');
  shell.cd(pkg);
  // Yarn above the other checks to make sure yarn doesn't change the lock file.
  $('yarn');
  console.log();

  console.log('~~~ Build npm ~~~');
  $('yarn build-npm for-publish');
  console.log();

  console.log('~~~ Tag version ~~~');
  shell.cd('..');
  $(`./scripts/tag-version.js ${pkg}`);
  console.log();

  console.log('~~~ Publishing to npm ~~~');
  shell.cd(pkg);
  $('npm publish');
  console.log();

  console.log(chalk.green('Yay! Published a new package to npm.'));
}

main();

