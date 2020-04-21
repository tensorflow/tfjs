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
 * This script should only be used for hot fix 1.7.x.
 * Steps for hot fix 1.7.x.
 * 1. Create a new release branch from the latest 1.7.x release branch. For
 * example, if the latest release branch is tfjs_1.7.2, create a new release
 * branch tfjs_1.7.3 from the latest commit in tfjs_1.7.2.
 *
 * 2. Fix the bug. Get PR approved and merged.
 *
 * 3. Use this script to prepare the release packages, starting from phase0.
 * You should still follow all the steps for release and publish.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as fs from 'fs';
import * as shell from 'shelljs';
import {RELEASE_UNITS, WEBSITE_RELEASE_UNIT, TMP_DIR, $, question, printReleaseUnit, printPhase, makeReleaseDir, updateDependency, prepareReleaseBuild, createPR} from './release-util';
import {releaseWebsite} from './release-website';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

// Computes the default updated version (does a patch version update).
function getPatchUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  return [versionSplit[0], versionSplit[1], +versionSplit[2] + 1].join('.');
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

  const releaseUnit = RELEASE_UNITS[releaseUnitInt];
  const {name, phases, repo} = releaseUnit;

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

  const phase = phases[phaseInt];
  const packages = phases[phaseInt].packages;
  const deps = phases[phaseInt].deps || [];

  if (releaseUnit === WEBSITE_RELEASE_UNIT) {
    await releaseWebsite(args);

    console.log(
        'Done. Please remember to deploy the website once you merged the PR.');

    process.exit(0);
  }

  // Release packages in tfjs repo.
  const dir = `${TMP_DIR}/tfjs`;
  makeReleaseDir(dir);

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  let releaseBranch = '';

  if (phaseInt !== 0) {
    // Phase0 should be published and release branch should have been created.
    const firstPackageLatestVersion =
        $(`npm view @tensorflow/${phases[0].packages[0]} dist-tags.latest`);
    releaseBranch = `${name}_${firstPackageLatestVersion}`;

    $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${
        dir} --depth=1`);
    shell.cd(dir);
  } else {
    // Phase0 needs user input of the release branch.
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

    $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${
        dir} --depth=1`);
    shell.cd(dir);
  }

  const newVersions = [];
  for (let i = 0; i < packages.length; i++) {
    const packageName = packages[i];
    shell.cd(packageName);

    // Update the version.
    const packageJsonPath = `${dir}/${packageName}/package.json`;
    let pkg = `${fs.readFileSync(packageJsonPath)}`;
    const parsedPkg = JSON.parse(`${pkg}`);
    const latestVersion =
        $(`npm view @tensorflow/${packageName} dist-tags.latest`);

    console.log(chalk.magenta.bold(
        `~~~ Processing ${packageName} (${latestVersion}) ~~~`));

    const patchUpdateVersion = getPatchUpdateVersion(latestVersion);
    let newVersion = latestVersion;
    newVersion =
        await question(`New version (leave empty for ${patchUpdateVersion}): `);
    if (newVersion === '') {
      newVersion = patchUpdateVersion;
    }

    // This condition should not happen.
    if (releaseBranch === '') {
      releaseBranch = `${name}_${newVersion}`;
      console.log(chalk.magenta.bold(
          `~~~ Creating new release branch ${releaseBranch} ~~~`));
      $(`git checkout -b ${releaseBranch}`);
      $(`git push origin ${releaseBranch}`);
    }

    pkg = `${pkg}`.replace(
        `"version": "${parsedPkg.version}"`, `"version": "${newVersion}"`);

    pkg = await updateDependency(deps, pkg, parsedPkg);

    fs.writeFileSync(packageJsonPath, pkg);

    prepareReleaseBuild(phase, packageName);

    shell.cd('..');

    $(`./scripts/make-version.js ${packageName}`);

    newVersions.push(newVersion);
  }

  const packageNames = packages.join(', ');
  const versionNames = newVersions.join(', ');
  const devBranchName = `dev_${releaseBranch}_phase${phaseInt}`;

  const message = `Update ${packageNames} to ${versionNames}.`;
  createPR(devBranchName, releaseBranch, message);

  console.log(
      `Done. FYI, this script does not publish to NPM. ` +
      `Please publish by running yarn publish-npm ` +
      `from each repo after you merge the PR.` +
      `Please remeber to update the website once you have released ` +
      'a new package version');

  process.exit(0);
}

main();
