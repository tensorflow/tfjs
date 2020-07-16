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
import {TMP_DIR, $, question, makeReleaseDir, createPR, TFJS_RELEASE_UNIT, updateMonorepoDependency} from './release-util';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

// Computes the default updated version (does a minor version update).
function getMinorUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  return [versionSplit[0], +versionSplit[1] + 1, versionSplit[2]].join('.');
}

async function main() {
  const args = parser.parseArgs();
  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  const dir = `${TMP_DIR}/tfjs`;
  makeReleaseDir(dir);

  // Guess release version from tfjs-core's latest version, with a minor update.
  const latestVersion =
  $(`npm view @tensorflow/tfjs-core dist-tags.latest`);
  const minorUpdateVersion = getMinorUpdateVersion(latestVersion);
  let newVersion = minorUpdateVersion;
  newVersion =
  await question(`New version (leave empty for ${minorUpdateVersion}): `);
  if (newVersion === '') {
    newVersion = minorUpdateVersion;
  }

  // Create a release branch in remote.
  $(`git clone ${urlBase}tensorflow/tfjs ${dir} --depth=1`);
  shell.cd(dir);
  const releaseBranch = `tfjs_${newVersion}`;
  console.log(chalk.magenta.bold(
    `~~~ Creating new release branch ${releaseBranch} ~~~`));
  $(`git checkout -b ${releaseBranch}`);
  $(`git push origin ${releaseBranch}`);

  // Update version.
  const phases = TFJS_RELEASE_UNIT.phases;

  for (let i = 0; i < phases.length; i++) {
    const packages = phases[i].packages;
    const deps = phases[i].deps || [];

    for (let i = 0; i < packages.length; i++) {
      const packageName = packages[i];
      shell.cd(packageName);

      // Update the version.
      const packageJsonPath = `${dir}/${packageName}/package.json`;
      let pkg = `${fs.readFileSync(packageJsonPath)}`;
      const parsedPkg = JSON.parse(`${pkg}`);

      console.log(chalk.magenta.bold(
          `~~~ Processing ${packageName} ~~~`));
      pkg = `${pkg}`.replace(
          `"version": "${parsedPkg.version}"`, `"version": "${newVersion}"`);

      pkg = await updateMonorepoDependency(deps, pkg, parsedPkg, newVersion);

      fs.writeFileSync(packageJsonPath, pkg);

      shell.cd('..');

      // Make version for all packages other than tfjs-node-gpu. Consider
      // remove make version.
      if (packageName !== 'tfjs-node-gpu') {
        $(`./scripts/make-version.js ${packageName}`);
      }
    }
  }

  // Use dev prefix to avoid branch being locked, delete the dev branch after
  // PR gets merged.
  const devBranchName = `dev_${releaseBranch}`;

  const message = `Update monorepo to ${newVersion}.`;
  createPR(devBranchName, releaseBranch, message);

  console.log(
      `Done. FYI, this script does not publish to NPM. ` +
      `Please publish by running  ` +
      `YARN_REGISTRY="https://registry.npmjs.org/" yarn publish-npm ` +
      `after you merge the PR.` +
      `Please remeber to update the website once you have released ` +
      'a new package version');

  process.exit(0);
}

main();
