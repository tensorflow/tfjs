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
 * This script creates pull requests to make releases for all the TensorFlow.js
 * packages.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as fs from 'fs';
import * as shell from 'shelljs';
import {TMP_DIR, $, question, makeReleaseDir, createPR, TFJS_RELEASE_UNIT, updateTFJSDependencyVersions, ALPHA_RELEASE_UNIT, getMinorUpdateVersion, getPatchUpdateVersion, E2E_PHASE} from './release-util';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
});

parser.addArgument('--local', {
  action: 'storeTrue',
  help: 'Only create the release branch locally. Do not push or create a PR.',
});

async function main() {
  const args = parser.parseArgs();
  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  const dir = `${TMP_DIR}/tfjs`;
  makeReleaseDir(dir);

  // Guess release version from tfjs-core's latest version, with a minor update.
  const latestVersion = $(`npm view @tensorflow/tfjs-core dist-tags.latest`);
  const minorUpdateVersion = getMinorUpdateVersion(latestVersion);
  const newVersion = await question('New version for monorepo (leave empty for '
    + `${minorUpdateVersion}): `) || minorUpdateVersion;

  // Populate the versions map with new versions for monorepo packages.
  const versions = new Map<string /* package name */, string /* version */>();
  for (const phase of TFJS_RELEASE_UNIT.phases) {
    for (const packageName of phase.packages) {
      versions.set(packageName, newVersion);
    }
  }

  // Add versions for alpha monorepo packages, which do not have the same
  // version as the other monorepo packages.
  for (const phase of ALPHA_RELEASE_UNIT.phases) {
    for (const packageName of phase.packages) {
      const latestVersion =
        $(`npm view @tensorflow/${packageName} dist-tags.latest`);
      const minorUpdateVersion = getPatchUpdateVersion(latestVersion);
      const newVersion =
        await question(`New version for alpha package ${packageName}`
          + ` (leave empty for ${minorUpdateVersion}): `) || minorUpdateVersion;
      versions.set(packageName, newVersion);
    }
  }

  // Get release candidate commit.
  const commit = await question(
      'Commit of release candidate (the last ' +
      'successful nightly build): ');
  if (commit === '') {
    console.log(chalk.red('Commit cannot be empty.'));
    process.exit(1);
  }

  // Create a release branch in remote.
  $(`git clone ${urlBase}tensorflow/tfjs ${dir}`);
  shell.cd(dir);
  const releaseBranch = `tfjs_${newVersion}`;
  console.log(chalk.magenta.bold(
      `~~~ Creating new release branch ${releaseBranch} ~~~`));
  $(`git checkout -b ${releaseBranch} ${commit}`);
  if (!args.local) {
    $(`git push origin ${releaseBranch}`);
  }

  // Update versions in package.json files.
  const phases = [
    ...TFJS_RELEASE_UNIT.phases, ...ALPHA_RELEASE_UNIT.phases, E2E_PHASE
  ];
  for (const phase of phases) {
    for (const packageName of phase.packages) {
      shell.cd(packageName);

      // Update the version.
      const packageJsonPath = `${dir}/${packageName}/package.json`;
      let pkg = `${fs.readFileSync(packageJsonPath)}`;
      const parsedPkg = JSON.parse(`${pkg}`);

      console.log(chalk.magenta.bold(`~~~ Processing ${packageName} ~~~`));
      const newVersion = versions.get(packageName);
      pkg = `${pkg}`.replace(
        `"version": "${parsedPkg.version}"`, `"version": "${newVersion}"`);
      pkg = updateTFJSDependencyVersions(pkg, versions, phase.deps || []);

      fs.writeFileSync(packageJsonPath, pkg);

      shell.cd('..');

      // Make version for all packages other than tfjs-node-gpu and e2e.
      if (packageName !== 'tfjs-node-gpu' && packageName !== 'e2e') {
        $(`./scripts/make-version.js ${packageName}`);
      }
    }
  }

  // Use dev prefix to avoid branch being locked.
  const devBranchName = `dev_${releaseBranch}`;

  const message = `Update monorepo to ${newVersion}.`;
  if (!args.local) {
    createPR(devBranchName, releaseBranch, message);
  }

  console.log(
      'Done. FYI, this script does not publish to NPM. ' +
      'Please publish by running  ' +
      'YARN_REGISTRY="https://registry.npmjs.org/" yarn publish-npm ' +
      'after you merge the PR.' +
      'Remember to delete the dev branch once PR is merged.' +
      'Please remeber to update the website once you have released ' +
      'a new package version.');

  if (args.local) {
    console.log(`Local output located in ${dir}`)
  }
  process.exit(0);
}

main();
