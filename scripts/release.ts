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
 * packages. The release process is split up into multiple phases. Each
 * phase will update the version of a package and the dependency versions and
 * send a pull request. Once the pull request is merged, you must publish the
 * packages manually from the individual packages.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as fs from 'fs';
import * as shell from 'shelljs';
import {RELEASE_UNITS, WEBSITE_RELEASE_UNIT, TMP_DIR, $, question, printReleaseUnit, printPhase, makeReleaseDir, updateDependency, prepareReleaseBuild, createPR, getPatchUpdateVersion, ALPHA_RELEASE_UNIT} from './release-util';
import {releaseWebsite} from './release-website';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

async function main() {
  const args = parser.parseArgs();

  // The alpha release unit is released with the monorepo and should not be
  // released by this script. Packages in the alpha release unit need their
  // package.json dependencies rewritten.
  const releaseUnits = RELEASE_UNITS.filter(r => r !== ALPHA_RELEASE_UNIT);
  releaseUnits.forEach(printReleaseUnit);
  console.log();

  const releaseUnitStr =
      await question('Which release unit (leave empty for 0): ');
  const releaseUnitInt = +releaseUnitStr;
  if (releaseUnitInt < 0 || releaseUnitInt >= releaseUnits.length) {
    console.log(chalk.red(`Invalid release unit: ${releaseUnitStr}`));
    process.exit(1);
  }
  console.log(chalk.blue(`Using release unit ${releaseUnitInt}`));
  console.log();

  const releaseUnit = releaseUnits[releaseUnitInt];
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
    // Phase0 needs user input of the release version to create release
    // branch.
    $(`git clone ${urlBase}tensorflow/tfjs ${dir} --depth=1`);
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
      `Please publish by running  ` +
      `YARN_REGISTRY="https://registry.npmjs.org/" yarn publish-npm ` +
      `after you merge the PR.` +
      `Please remeber to update the website once you have released ` +
      'a new package version');

  process.exit(0);
}

main();
