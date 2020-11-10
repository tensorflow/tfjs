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

import chalk from 'chalk';
import * as mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as shell from 'shelljs';

export interface Phase {
  // The list of packages that will be updated with this change.
  packages: string[];
  // The list of dependencies that all of the packages will update to.
  deps?: string[];
  // An ordered map of scripts, key is package name, value is an object with two
  // optional fields: `before-yarn` with scripts to run before `yarn`, and
  // `after-yarn` with scripts to run after yarn is called and before the pull
  // request is sent out.
  scripts?: {[key: string]: {[key: string]: string[]}};
  // Whether to leave the version of the package alone. Defaults to false
  // (change the version).
  leaveVersion?: boolean;
  title?: string;
}

export interface ReleaseUnit {
  // A human-readable name. Used for generating release branch.
  name: string;
  // The phases in this release unit.
  phases: Phase[];
  // The repository *only if it is not the same as tfjs*.
  repo?: string;
}

export const CORE_PHASE: Phase = {
  packages: ['tfjs-core'],
  // Do not mark tfjs-backend-cpu as a dependency during releases. As a
  // devDependency it should keep the link:// path. Once tests have passed in CI
  // building and releasing core should not depend on the cpu backend
};

export const CPU_PHASE: Phase = {
  packages: ['tfjs-backend-cpu'],
  deps: ['tfjs-core']
};

export const WEBGL_PHASE: Phase = {
  packages: ['tfjs-backend-webgl'],
  deps: ['tfjs-core', 'tfjs-backend-cpu']
};

export const LAYERS_CONVERTER_PHASE: Phase = {
  packages: ['tfjs-layers', 'tfjs-converter'],
  deps: ['tfjs-core', 'tfjs-backend-cpu', 'tfjs-backend-webgl']
};

export const DATA_PHASE: Phase = {
  packages: ['tfjs-data'],
  deps: ['tfjs-core', 'tfjs-layers', 'tfjs-backend-cpu']
}

export const UNION_PHASE: Phase = {
  packages: ['tfjs'],
  deps: [
    'tfjs-core', 'tfjs-layers', 'tfjs-converter', 'tfjs-data',
    'tfjs-backend-cpu', 'tfjs-backend-webgl'
  ]
};

// We added tfjs-core and tfjs-layers because Node has unit tests that directly
// use tf.core and tf.layers to test serialization of models. Consider moving
// the test to tf.layers.
export const NODE_PHASE: Phase = {
  packages: ['tfjs-node', 'tfjs-node-gpu'],
  deps: ['tfjs', 'tfjs-core'],
  scripts: {'tfjs-node-gpu': {'before-yarn': ['yarn prep-gpu']}}
};

export const WASM_PHASE: Phase = {
  packages: ['tfjs-backend-wasm'],
  deps: ['tfjs-core', 'tfjs-backend-cpu']
};

export const VIS_PHASE: Phase = {
  packages: ['tfjs-vis']
};

export const REACT_NATIVE_PHASE: Phase = {
  packages: ['tfjs-react-native'],
  deps: ['tfjs-core', 'tfjs-backend-cpu', 'tfjs-backend-webgl']
};

export const WEBSITE_PHASE: Phase = {
  packages: ['tfjs-website'],
  deps: ['tfjs', 'tfjs-node', 'tfjs-vis', 'tfjs-react-native'],
  scripts: {'tfjs-website': {'after-yarn': ['yarn prep && yarn build-prod']}},
  leaveVersion: true,
  title: 'Update website to latest dependencies.'
};

export const TFJS_RELEASE_UNIT: ReleaseUnit = {
  name: 'tfjs',
  phases: [
    CORE_PHASE, CPU_PHASE, WEBGL_PHASE, LAYERS_CONVERTER_PHASE, DATA_PHASE,
    UNION_PHASE, NODE_PHASE, WASM_PHASE
  ]
};

export const VIS_RELEASE_UNIT: ReleaseUnit = {
  name: 'vis',
  phases: [VIS_PHASE]
};

export const REACT_NATIVE_RELEASE_UNIT: ReleaseUnit = {
  name: 'react-native',
  phases: [REACT_NATIVE_PHASE]
};

export const WEBSITE_RELEASE_UNIT: ReleaseUnit = {
  name: 'website',
  phases: [WEBSITE_PHASE],
  repo: 'tfjs-website'
};

export const RELEASE_UNITS: ReleaseUnit[] = [
  TFJS_RELEASE_UNIT, VIS_RELEASE_UNIT, REACT_NATIVE_RELEASE_UNIT,
  WEBSITE_RELEASE_UNIT
];

export const TMP_DIR = '/tmp/tfjs-release';

const rl =
    readline.createInterface({input: process.stdin, output: process.stdout});

export async function question(questionStr: string): Promise<string> {
  console.log(chalk.bold(questionStr));
  return new Promise<string>(
      resolve => rl.question('> ', response => resolve(response)));
}

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
export function $(cmd: string) {
  const result = shell.exec(cmd, {silent: true});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}

export function printReleaseUnit(id: number) {
  const releaseUnit = RELEASE_UNITS[id];
  console.log(chalk.green(`Release unit ${id}:`));
  console.log(` packages: ${
      chalk.blue(releaseUnit.phases.map(phase => phase.packages.join(', '))
                     .join(', '))}`);
}

export function printPhase(phases: Phase[], phaseId: number) {
  const phase = phases[phaseId];
  console.log(chalk.green(`Phase ${phaseId}:`));
  console.log(`  packages: ${chalk.blue(phase.packages.join(', '))}`);
  if (phase.deps != null) {
    console.log(`   deps: ${phase.deps.join(', ')}`);
  }
}

export function makeReleaseDir(dir: string) {
  mkdirp(TMP_DIR, err => {
    if (err) {
      console.log('Error creating temp dir', TMP_DIR);
      process.exit(1);
    }
  });
  $(`rm -f -r ${dir}/*`);
  $(`rm -f -r ${dir}`);
  $(`mkdir ${dir}`);
}

export async function updateDependency(
    deps: string[], pkg: string, parsedPkg: any): Promise<string> {
  console.log(chalk.magenta.bold(`~~~ Update dependency versions ~~~`));

  if (deps != null) {
    const depsLatestVersion: string[] =
        deps.map(dep => $(`npm view @tensorflow/${dep} dist-tags.latest`));

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

  return pkg;
}

// Update package.json dependencies of tfjs packages. This method is different
// than `updateDependency`, it does not rely on published versions, instead it
// assumes all the packages have the same version and use that to update.
export function updateTFJSDependencyVersions(
    deps: string[], pkg: string, parsedPkg: any, tfjsVersion: string): string {
  console.log(chalk.magenta.bold(`~~~ Update dependency versions ~~~`));

  if (deps != null) {
    for (let j = 0; j < deps.length; j++) {
      const dep = deps[j];

      // Get the current dependency package version.
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
      const versionLatest = relaxedVersionPrefix + tfjsVersion;

      pkg = `${pkg}`.replace(
          new RegExp(`"${depNpmName}": "${version}"`, 'g'),
          `"${depNpmName}": "${versionLatest}"`);
    }
  }

  return pkg;
}

export function prepareReleaseBuild(phase: Phase, packageName: string) {
  console.log(chalk.magenta.bold(`~~~ Prepare release build ~~~`));
  console.log(chalk.bold('Prepare before-yarn'));
  if (phase.scripts != null && phase.scripts[packageName] != null &&
      phase.scripts[packageName]['before-yarn'] != null) {
    phase.scripts[packageName]['before-yarn'].forEach(script => $(script));
  }

  console.log(chalk.bold('yarn'));
  $(`yarn`);

  console.log(chalk.bold('Prepare after-yarn'));
  if (phase.scripts != null && phase.scripts[packageName] != null &&
      phase.scripts[packageName]['after-yarn'] != null) {
    phase.scripts[packageName]['after-yarn'].forEach(script => $(script));
  }
}

export async function getReleaseBranch(name: string): Promise<string> {
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
  releaseBranch = await question(
      `Which release branch (leave empty for ` +
      `${maybeBranch}):`);
  if (releaseBranch === '') {
    releaseBranch = maybeBranch;
  }

  return releaseBranch;
}

export function checkoutReleaseBranch(
    releaseBranch: string, git_protocol: string, dir: string) {
  console.log(chalk.magenta.bold(
      `~~~ Checking out release branch ${releaseBranch} ~~~`));
  $(`rm -f -r ${dir}`);
  mkdirp(dir, err => {
    if (err) {
      console.log('Error creating temp dir', dir);
      process.exit(1);
    }
  });

  const urlBase = git_protocol ? 'git@github.com:' : 'https://github.com/';
  $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${dir} --depth=1`);
}

export function createPR(
    devBranchName: string, releaseBranch: string, message: string) {
  console.log(
      chalk.magenta.bold('~~~ Creating PR to update release branch ~~~'));
  $(`git checkout -b ${devBranchName}`);
  $(`git push -u origin ${devBranchName}`);
  $(`git add .`);
  $(`git commit -a -m "${message}"`);
  $(`git push`);

  $(`hub pull-request -b ${releaseBranch} -m "${message}" -l INTERNAL -o`);
  console.log();
}
