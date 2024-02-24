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
import semver from 'semver';
import * as fs from 'fs';
import * as shell from 'shelljs';
import {TMP_DIR, $, question, makeReleaseDir, createPR, TFJS_RELEASE_UNIT, updateTFJSDependencyVersions, ALPHA_RELEASE_UNIT, getMinorUpdateVersion, getPatchUpdateVersion, E2E_PHASE, getReleaseBlockers, getNightlyVersion} from './release-util';
import * as path from 'path';
import {findDeps} from './graph_utils';

const parser = new argparse.ArgumentParser({
  description: 'Create a release PR for the tfjs monorepo.',
});

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
});

parser.addArgument(['--dry'], {
  action: 'storeTrue',
  help: 'Only create the release branch locally. Do not push or create a PR.',
});

parser.addArgument('--guess-version', {
  type: 'string',
  choices: ['release', 'nightly'],
  help: 'Use the guessed version without asking for confirmation.',
});

parser.addArgument(['--commit-hash', '--hash'], {
  type: 'string',
  help: 'Commit hash to publish. Usually the latest successful nightly run.',
});

parser.addArgument(['--use-local-changes'], {
  action: 'storeTrue',
  help: 'Use local changes to the repo instead of a remote branch. Only for' +
      ' testing and debugging.',
});

parser.addArgument('--force', {
  action: 'storeTrue',
  help: 'Force a release even if there are release blockers.',
});

async function getNewVersion(
    packageName: string, incrementVersion: (version: string) => string,
    ask = true) {
  let newVersion: string|undefined;
  try {
    const versions: string[] =
        JSON.parse($(`npm view @tensorflow/${packageName} versions --json`));
    if (Array.isArray(versions) && versions.length !== 0) {
      const latestVersion = semver.rsort(versions)[0];
      newVersion = incrementVersion(latestVersion);
    }
  } catch (e) {
    // Suppress errors when guessing the version.
  }

  if (newVersion != null) {
    if (!ask) {
      return newVersion;
    }
    newVersion = await question(`New version for ${
                     packageName} (leave empty for ${newVersion}): `) ||
        newVersion;
    return newVersion;
  }

  if (!ask) {
    console.warn(
        'Guessing version 0.0.1 for unpublished package ' +
        `${packageName}`);
    return '0.0.1';
  }

  // Repeat until the user answers.
  while (true) {
    newVersion = await question(
        `New Version for ${packageName} (no current version found on npm): `);
    if (newVersion !== '') {
      return newVersion;
    }
    console.log(
        `${packageName} has no version on npm. ` +
        'Please provide an initial version.');
  }
}

async function main() {
  const args = parser.parseArgs();

  let incrementVersion: ((version: string) => string)|undefined;
  if (args.guess_version === 'nightly') {
    incrementVersion = v => getNightlyVersion(getMinorUpdateVersion(v));
  }

  if (args.use_local_changes) {
    // Force dry run when using local files instead of a release branch.
    // This is for debugging.
    args.dry = true;
  }
  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  const dir = `${TMP_DIR}/tfjs`;
  makeReleaseDir(dir);

  if (args.force) {
    console.warn('Ignoring any potential release blockerse due to \'--force\'');
  } else {
    const blockers = getReleaseBlockers();
    if (blockers) {
      throw new Error(`Can not release due to release blockers:\n ${blockers}`);
    }
  }

  // Guess release version from tfjs-core's latest version, with a minor update.
  const newVersion = await getNewVersion(
      'tfjs-core', incrementVersion ?? getMinorUpdateVersion,
      !args.guess_version);

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
      const newVersion = await getNewVersion(
          packageName, incrementVersion ?? getPatchUpdateVersion,
          !args.guess_version);
      versions.set(packageName, newVersion);
    }
  }

  // Get release candidate commit.
  let commit = args.commit_hash;
  if (!args.use_local_changes) {
    if (!commit) {
      commit = await question(
          'Commit of release candidate (the last ' +
          'successful nightly build): ');
    }
    if (commit === '') {
      console.log(chalk.red('Commit cannot be empty.'));
      process.exit(1);
    }
  }

  // Create a release branch in remote.
  $(`git clone ${urlBase}tensorflow/tfjs ${dir}`);

  const releaseBranch = `tfjs_${newVersion}`;

  if (args.use_local_changes) {
    shell.cd(path.join(__dirname, '../'));
    console.log(chalk.magenta.bold(
        '~~~ Copying current changes to a new release branch' +
        ` ${releaseBranch} ~~~`));
    // Avoid copying `.git/` because this script will `git push`
    // to origin, which it expects to be the tfjs repo as was set
    // up when the script ran 'git clone' above.
    // This makes sure other hidden files like .bazelrc are copied.
    $(`cp -r \`ls -A | grep -v ".git"\` ${dir}`);
    shell.cd(dir);
  } else {
    shell.cd(dir);
    console.log(chalk.magenta.bold(
        `~~~ Creating new release branch ${releaseBranch} ~~~`));
    $(`git checkout -b ${releaseBranch} ${commit}`);
  }
  if (!args.dry) {
    $(`git push origin ${releaseBranch}`);
  }

  // Update versions in package.json files.
  const phases =
      [...TFJS_RELEASE_UNIT.phases, ...ALPHA_RELEASE_UNIT.phases, E2E_PHASE];
  const errors: Error[] = [];
  for (const phase of phases) {
    for (const packageName of phase.packages) {
      shell.cd(packageName);

      // Update the version number of the package.json
      const packagePath = path.join(dir, packageName);
      const packageJsonPath = path.join(packagePath, 'package.json');
      let pkg = fs.readFileSync(packageJsonPath, 'utf8');
      const parsedPkg = JSON.parse(`${pkg}`);

      console.log(chalk.magenta.bold(`~~~ Processing ${packageName} ~~~`));
      const newVersion = versions.get(packageName);
      pkg = `${pkg}`.replace(
          `"version": "${parsedPkg.version}"`, `"version": "${newVersion}"`);

      fs.writeFileSync(packageJsonPath, pkg);

      // Update dependency versions of all package.json files found in the
      // package to use the new version numbers (except ones in node_modules).
      const subpackages =
          $(`find ${
                packagePath} -name package.json -not -path \'*/node_modules/*\'`)
              .split('\n');
      for (const packageJsonPath of subpackages) {
        const pkg = fs.readFileSync(packageJsonPath, 'utf8');
        console.log(chalk.magenta.bold(
            `~~~ Update dependency versions for ${packageJsonPath} ~~~`));

        // Only update versions that are a (possibly transitive) dependency of
        // the package and are listed in the phase deps (we throw an error
        // if we find a dependency that doesn't satisfy these conditions).
        const transitiveDeps = [...findDeps([packageName])].filter(
            dep => phase.deps.includes(dep));

        // Also add the package itself so subpackages can use it.
        // Some packages, like e2e, are never published to npm, so check first.
        if (versions.has(packageName)) {
          transitiveDeps.push(packageName);
        }

        const packageDependencyVersions =
            new Map(transitiveDeps.map(dep => [dep, versions.get(dep)!]));

        try {
          const updated =
              updateTFJSDependencyVersions(pkg, packageDependencyVersions);

          fs.writeFileSync(packageJsonPath, updated);
        } catch (e) {
          e.message = `For ${packageJsonPath}, ${packageName} ${e.message}`;
          console.error(e.stack);
          errors.push(e);
        }
      }

      shell.cd('..');

      // Make version for all packages other than tfjs-node-gpu and e2e.
      if (packageName !== 'tfjs-node-gpu' && packageName !== 'e2e') {
        $(`./scripts/make-version.js ${packageName}`);
      }
    }
  }
  if (errors.length > 0) {
    throw new Error('Some package version updates had errors' + errors);
  }


  // Use dev prefix to avoid branch being locked.
  const devBranchName = `dev_${releaseBranch}`;

  const message = `Update monorepo to ${newVersion}.`;
  if (!args.dry) {
    createPR(devBranchName, releaseBranch, message);
  }

  console.log(
      'Done. FYI, this script does not publish to NPM. ' +
      'Please publish by running  ' +
      'YARN_REGISTRY="https://registry.npmjs.org/" yarn publish-npm ' +
      'after you merge the PR.' +
      'Remember to delete the dev branch once PR is merged.' +
      'Please remember to update the website once you have released ' +
      'a new package version.');

  if (args.dry) {
    console.log(`No PR was created. Local output is located in ${dir}.`);
  }
  process.exit(0);
}

main();
