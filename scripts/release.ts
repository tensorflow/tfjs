#!/usr/bin/env node
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as shell from 'shelljs';
import {RELEASE_UNITS, Phase, $, question, printReleaseUnit} from './release-util';

const TMP_DIR = '/tmp/tfjs-release';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

function printPhase(phases: Phase[], phaseId: number) {
  const phase = phases[phaseId];
  console.log(chalk.green(`Phase ${phaseId}:`));
  console.log(`  packages: ${chalk.blue(phase.packages.join(', '))}`);
  if (phase.deps != null) {
    console.log(`   deps: ${phase.deps.join(', ')}`);
  }
}

// Computes the default updated version (does a patch version update).
function getPatchUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  return [versionSplit[0], versionSplit[1], +versionSplit[2] + 1].join('.');
}

function publishTFJS(repo: string): boolean {
  return repo == null;
}

async function main() {
  const args = parser.parseArgs();

  RELEASE_UNITS.forEach((_, i) => printReleaseUnit(i));
  console.log();

  const releaseUnitStr = await question('Which release unit (leave empty for 0): ');
  const releaseUnitInt = +releaseUnitStr;
  if (releaseUnitInt < 0 || releaseUnitInt >= RELEASE_UNITS.length) {
    console.log(chalk.red(`Invalid release unit: ${releaseUnitStr}`));
    process.exit(1);
  }
  console.log(chalk.blue(`Using release unit ${releaseUnitInt}`));
  console.log();

  const {name, phases, repo} = RELEASE_UNITS[releaseUnitInt];

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

  const dir = `${TMP_DIR}/${publishTFJS(repo) ? `tfjs` : repo}`;
  mkdirp(TMP_DIR, err => {
    if (err) {
      console.log('Error creating temp dir', TMP_DIR);
      process.exit(1);
    }
  });
  $(`rm -f -r ${dir}/*`);
  $(`rm -f -r ${dir}`);
  $(`mkdir ${dir}`);

  const dirMaster = `${TMP_DIR}/${publishTFJS(repo) ? `tfjs_master` : repo}`;

  // Only generate a second folder for master branch if publish tfjs.
  if (publishTFJS(repo)) {
    $(`rm -f -r ${dirMaster}/*`);
    $(`rm -f -r ${dirMaster}`);
    $(`mkdir ${dirMaster}`);
  }

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  let releaseBranch = '';

  if (repo != null) {
    // Publishing website, another repo.
    $(`git clone ${urlBase}tensorflow/${repo} ${dir} --depth=1`);
    shell.cd(dir);
  } else {
    // Publishing packages in tfjs.
    // For monorepo, need to update both release branch and master branch.
    // Master branch will have a simpler update, just bump version and
    // update version.ts. Release branch also have to update dependencies.
    $(`git clone -b master ${urlBase}tensorflow/tfjs ${dirMaster} --depth=1`);

    if (phaseInt !== 0) {
      // Phase0 should be published and release branch should have been created.
      const latestVersion =
        $(`npm view @tensorflow/${phases[0].packages[0]} dist-tags.latest`);
      releaseBranch = `${name}_${latestVersion}`;

      $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${
        dir} --depth=1`);
      shell.cd(dir);
    } else {
      // Phase0 needs user input of the release version to create release
      // branch.
      $(`git clone ${urlBase}tensorflow/tfjs ${dir} --depth=1`);
      shell.cd(dir);
    }
  }

  const newVersions = [];
  for (let i = 0; i < packages.length; i++) {
    const packageName = packages[i];
    if (publishTFJS(repo)) {
      shell.cd(packageName);
    }

    const depsLatestVersion: string[] =
      deps.map(dep => $(`npm view @tensorflow/${dep} dist-tags.latest`));

    // Update the version.
    const packageJsonPath = publishTFJS(repo) ?
      `${dir}/${packageName}/package.json` :
      `${dir}/package.json`;
    let pkg = `${fs.readFileSync(packageJsonPath)}`;
    const parsedPkg = JSON.parse(`${pkg}`);

    console.log(chalk.magenta.bold(
      `~~~ Processing ${packageName} (${parsedPkg.version}) ~~~`));

    const patchUpdateVersion = getPatchUpdateVersion(parsedPkg.version);
    let newVersion = parsedPkg.version;
    if (!phase.leaveVersion) {
      newVersion = await question(
        `New version (leave empty for ${patchUpdateVersion}): `);
      if (newVersion === '') {
        newVersion = patchUpdateVersion;
      }
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

    if (deps != null) {
      for (let j = 0; j < deps.length; j++) {
        const dep = deps[j];

        let version = '';
        const depNpmName = `@tensorflow/${dep}`;
        if (parsedPkg['dependencies'] != null &&
          parsedPkg['dependencies'][depNpmName] != null) {
          version = parsedPkg['dependencies'][depNpmName];
        } else if (
          parsedPkg['peerDependencies'] != null &&
          parsedPkg['peerDependencies'][depNpmName] != null) {
          version = parsedPkg['peerDependencies'][depNpmName];
        } else if (
          parsedPkg['devDependencies'] != null &&
          parsedPkg['devDependencies'][depNpmName] != null) {
          version = parsedPkg['devDependencies'][depNpmName];
        }
        if (version == null) {
          throw new Error(`No dependency found for ${dep}.`);
        }

        let relaxedVersionPrefix = '';
        if (version.startsWith('~') || version.startsWith('^')) {
          relaxedVersionPrefix = version.substr(0, 1);
        }
        const depVersionLatest = relaxedVersionPrefix + depsLatestVersion[j];

        let depVersion = await question(
          `Updated version for ` +
          `${dep} (current is ${version}, leave empty for latest ${
          depVersionLatest}): `);
        if (depVersion === '') {
          depVersion = depVersionLatest;
        }
        console.log(chalk.blue(`Using version ${depVersion}`));

        pkg = `${pkg}`.replace(
          new RegExp(`"${depNpmName}": "${version}"`, 'g'),
          `"${depNpmName}": "${depVersion}"`);
      }
    }

    fs.writeFileSync(packageJsonPath, pkg);
    if (phase.scripts != null && phase.scripts[packageName] != null &&
      phase.scripts[packageName]['before-yarn'] != null) {
      phase.scripts[packageName]['before-yarn'].forEach(script => $(script));
    }
    $(`yarn`);
    if (phase.scripts != null && phase.scripts[packageName] != null &&
      phase.scripts[packageName]['after-yarn'] != null) {
      phase.scripts[packageName]['after-yarn'].forEach(script => $(script));
    }
    if (publishTFJS(repo)) {
      shell.cd('..');
    }
    if (!phase.leaveVersion) {
      $(`./scripts/make-version.js ${packageName}`);
    }
    newVersions.push(newVersion);

    // For master branch, only update package.json version.
    if (publishTFJS(repo)) {
      shell.cd(dirMaster);

      const packageJsonPathMaster = `${dirMaster}/${packageName}/package.json`;
      fs.writeFileSync(packageJsonPathMaster, pkg);

      if (!phase.leaveVersion) {
        $(`./scripts/make-version.js ${packageName}`);
      }

      shell.cd(dir);
    }
  }

  const packageNames = packages.join(', ');
  const versionNames = newVersions.join(', ');
  const branchName = `${releaseBranch}_phase${phaseInt}`;

  console.log(chalk.magenta.bold(
    '~~~ Creating PR to update release branch ~~~'));
  $(`git checkout -b ${branchName}`);
  $(`git push -u origin ${branchName}`);
  $(`git add .`);
  $(`git commit -a -m "Update ${packageNames} to ${versionNames}."`);
  $(`git push`);
  const title =
    phase.title ? phase.title : `Update ${packageNames} to ${versionNames}.`;
  $(`hub pull-request -b ${releaseBranch} -m "${title}" -l INTERNAL -o`);
  console.log();

  if (publishTFJS(repo)) {
    console.log(chalk.magenta.bold(
      '~~~ Creating PR to update master branch ~~~'));
    console.log(chalk.blue(`You should NOT merge this PR before the PR for
      release branch is merged.`));
    const branchNameMaster = `${branchName}_master`;
    shell.cd(dirMaster);
    $(`git checkout -b ${branchNameMaster}`);
    $(`git push -u origin ${branchNameMaster}`);
    $(`git add .`);
    $(`git commit -a -m "Update ${packageNames} to ${versionNames}."`);
    $(`git push`);
    const title =
      phase.title ? phase.title : `Update ${packageNames} to ${versionNames}.`;
    $(`hub pull-request -m "${title}" -l INTERNAL -o`);
    console.log();
  }

  console.log(
    `Done. FYI, this script does not publish to NPM. ` +
    `Please publish by running ./scripts/publish-npm.sh ` +
    `from each repo after you merge the PR.` +
    `Please remeber to update the website once you have released ` +
    'a new package version');

  process.exit(0);
}

main();
