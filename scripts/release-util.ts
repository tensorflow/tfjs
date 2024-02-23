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
import * as fs from 'fs';
import * as inquirer from 'inquirer';
import {Separator} from 'inquirer';
import mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as shell from 'shelljs';
import rimraf from 'rimraf';
import * as path from 'path';
import {fork} from 'child_process';

export interface Phase {
  // The list of packages that will be updated with this change.
  packages: string[];
  // The list of dependencies that all of the packages will update to.
  // TODO(mattSoulanille): Parse this from package_dependencies.json or from the
  // package.json file of each package.
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

export const WEBGPU_PHASE: Phase = {
  packages: ['tfjs-backend-webgpu'],
  deps: ['tfjs-core', 'tfjs-backend-cpu'],
};

export const VIS_PHASE: Phase = {
  packages: ['tfjs-vis']
};

export const REACT_NATIVE_PHASE: Phase = {
  packages: ['tfjs-react-native'],
  deps: ['tfjs-core', 'tfjs-backend-cpu', 'tfjs-backend-webgl']
};

export const TFDF_PHASE: Phase = {
  packages: ['tfjs-tfdf'],
  deps: ['tfjs-core', 'tfjs-backend-cpu', 'tfjs-converter']
};

export const TFLITE_PHASE: Phase = {
  packages: ['tfjs-tflite'],
  deps: ['tfjs-core', 'tfjs-backend-cpu']
};

export const AUTOML_PHASE: Phase = {
  packages: ['tfjs-automl'],
  deps: ['tfjs-core', 'tfjs-backend-webgl', 'tfjs-converter']
};

export const WEBSITE_PHASE: Phase = {
  packages: ['tfjs-website'],
  deps: [
    'tfjs', 'tfjs-node', 'tfjs-vis', 'tfjs-react-native', 'tfjs-tfdf',
    'tfjs-tflite', '@tensorflow-models/tasks'
  ],
  scripts: {'tfjs-website': {'after-yarn': ['yarn prep && yarn build-prod']}},
  leaveVersion: true,
  title: 'Update website to latest dependencies.'
};

// Note that e2e is not actually published. As a result, this phase is not
// included in any release unit, however, it is used for updating dependencies.
export const E2E_PHASE: Phase = {
  packages: ['e2e'],
  deps: [
    'tfjs', 'tfjs-backend-cpu', 'tfjs-backend-wasm', 'tfjs-backend-webgl',
    'tfjs-backend-webgpu', 'tfjs-converter', 'tfjs-core', 'tfjs-data',
    'tfjs-layers', 'tfjs-node'
  ],
}

export const TFJS_RELEASE_UNIT: ReleaseUnit = {
  name: 'tfjs',
  phases: [
    CORE_PHASE, CPU_PHASE, WEBGL_PHASE, WEBGPU_PHASE, LAYERS_CONVERTER_PHASE,
    DATA_PHASE, UNION_PHASE, NODE_PHASE, WASM_PHASE
  ]
};

// TODO(mattsoulanille): Move WEBGPU_PHASE to TFJS_RELEASE_UNIT when webgpu
// is out of alpha.
// Alpha packages use monorepo dependencies at the latest version but are
// not yet released at the same version number as the monorepo packages.
// Use this for packages that will be a part of the monorepo in the future.
// The release script will ask for a new version for each phase, and it will
// replace 'link' dependencies with the new monorepo version.
export const ALPHA_RELEASE_UNIT: ReleaseUnit = {
  name: 'alpha-monorepo-packages',
  phases: [TFDF_PHASE],
};

export const VIS_RELEASE_UNIT: ReleaseUnit = {
  name: 'vis',
  phases: [VIS_PHASE]
};

export const REACT_NATIVE_RELEASE_UNIT: ReleaseUnit = {
  name: 'react-native',
  phases: [REACT_NATIVE_PHASE]
};

export const TFLITE_RELEASE_UNIT: ReleaseUnit = {
  name: 'tflite',
  phases: [TFLITE_PHASE]
};

export const AUTOML_RELEASE_UNIT: ReleaseUnit = {
  name: 'automl',
  phases: [AUTOML_PHASE]
};

export const WEBSITE_RELEASE_UNIT: ReleaseUnit = {
  name: 'website',
  phases: [WEBSITE_PHASE],
  repo: 'tfjs-website'
};

export const RELEASE_UNITS: ReleaseUnit[] = [
  TFJS_RELEASE_UNIT,
  ALPHA_RELEASE_UNIT,
  VIS_RELEASE_UNIT,
  REACT_NATIVE_RELEASE_UNIT,
  TFLITE_RELEASE_UNIT,
  AUTOML_RELEASE_UNIT,
  WEBSITE_RELEASE_UNIT,
];

export const ALL_PACKAGES: Set<string> = new Set(getPackages(RELEASE_UNITS));

export const TMP_DIR = '/tmp/tfjs-release';

export async function question(questionStr: string): Promise<string> {
  const rl =
    readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log(chalk.bold(questionStr));
  return new Promise<string>(
    resolve => {
      rl.question('> ', response => {
        resolve(response);
        rl.close();
      });
    });
}

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
export function $(cmd: string, env: Record<string, string> = {}) {
  env = {...process.env, ...env};

  const result = shell.exec(cmd, {silent: true, env});
  if (result.code > 0) {
    throw new Error(`$ ${cmd}\n ${result.stderr}`);
  }
  return result.stdout.trim();
}

/**
 * An async wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
export function $async(cmd: string,
                       env: Record<string, string> = {}): Promise<string> {
  env = {...shell.env, ...env};
  return new Promise((resolve, reject) => {
    shell.exec(cmd, {silent: true, env}, (code, stdout, stderr) => {
      if (code > 0) {
        console.log('$', cmd);
        console.log(stdout);
        console.log(stderr);
        reject(stderr);
      }
      resolve(stdout.trim());
    })
  });
}

export function printReleaseUnit(releaseUnit: ReleaseUnit, id: number) {
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
    const depsLatestVersion: string[] = deps.map(
        dep => $(`npm view ${
            dep.includes('@') ? dep : '@tensorflow/' + dep} dist-tags.latest`));

    for (let j = 0; j < deps.length; j++) {
      const dep = deps[j];

      let version = '';
      const depNpmName = dep.includes('@') ? dep : `@tensorflow/${dep}`;
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
        relaxedVersionPrefix = version.slice(0, 1);
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
// uses a map from packageName to newVersion to update the versions.
export function updateTFJSDependencyVersions(
    pkg: string, versions: Map<string, string>,
    depsToReplace = [...versions.keys()]): string {

  const parsedPkg = JSON.parse(pkg);

  const dependencyMaps: Array<{[index: string]: string}> = [
    parsedPkg['dependencies'],
    parsedPkg['peerDependencies'],
    parsedPkg['devDependencies'],
  ].filter(v => v != null);

  for (const dependencyMap of dependencyMaps) {
    for (const [name, version] of Object.entries(dependencyMap)) {
      const prefix = '@tensorflow/';
      if (name.startsWith(prefix) && version.startsWith('link:')) {
        const tfjsName = name.slice(prefix.length);
        const newVersion = versions.get(tfjsName);
        if (newVersion == null) {
          throw new Error(`Versions map does not include ${tfjsName}`);
        }

        let relaxedVersionPrefix = '';
        if (version.startsWith('~') || version.startsWith('^')) {
          relaxedVersionPrefix = version.slice(0, 1);
        }
        const versionLatest = relaxedVersionPrefix + newVersion;
        pkg = `${pkg}`.replace(
          new RegExp(`"${name}": "${version}"`, 'g'),
          `"${name}": "${versionLatest}"`);
      }
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
  // It will not match temporary generated branches such as tfjs_1.7.1_phase0.
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

/**
 * Get all GitHub issues tagged as release blockers.
 *
 * @return A string of all the issues. Empty if there are none.
 */
export function getReleaseBlockers() {
  return $('hub issue -l "RELEASE BLOCKER"');
}

// Computes the default updated version (does a patch version update).
export function getPatchUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  // For alpha or beta version string (e.g. "0.0.1-alpha.5"), increase the
  // number after alpha/beta.
  if (versionSplit[2].includes('alpha') || versionSplit[2].includes('beta')) {
    return [
      versionSplit[0], versionSplit[1], versionSplit[2], +versionSplit[3] + 1
    ].join('.');
  }

  return [versionSplit[0], versionSplit[1], +versionSplit[2] + 1].join('.');
}


/**
 * Get the next minor update version for the given version.
 *
 * e.g. given 1.2.3, return 1.3.0
 */
export function getMinorUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  return [versionSplit[0], + versionSplit[1] + 1, '0'].join('.');
}

/**
 * Create the nightly version string by appending `dev-{current date}` to the
 * given version.
 *
 * Versioning format is from semver: https://semver.org/spec/v2.0.0.html
 * This version should be published with the 'next' tag and should increment the
 * current 'latest' tfjs version.
 * We approximate TypeScript's versioning practice as seen on their npm page
 * https://www.npmjs.com/package/typescript?activeTab=versions
 */
export function getNightlyVersion(version: string): string {
  // Format date to YYYYMMDD.
  const date =
      new Date().toISOString().split('T')[0].replace(new RegExp('-', 'g'), '');
  return `${version}-dev.${date}`;
}

/**
 * Filter a list with an async filter function
 */
async function filterAsync<T>(
  array: T[],
  condition: (t: T) => Promise<boolean>): Promise<T[]> {

  const results = await Promise.all(array.map(condition));
  return array.filter((_val, index) => results[index]);
}

/**
 * Get the packages contained in the given release units.
 */
export function getPackages(releaseUnits: ReleaseUnit[]): string[] {
  return releaseUnits.map(releaseUnit => releaseUnit.phases)
    .flat().map(phase => phase.packages)
    .flat();
}

/**
 * Filter packages in release units according to an async filter.
 */
export async function filterPackages(filter: (pkg: string) => Promise<boolean>,
                                     releaseUnits = RELEASE_UNITS) {
  return filterAsync(getPackages(releaseUnits), filter);
}

export async function selectPackages({
  message = "Select packages",
  selected = async (_pkg: string) => false,
  modifyName = async (name: string) => name,
  releaseUnits = RELEASE_UNITS}) {

  type SeparatorInstance = InstanceType<typeof Separator>;
  type Choice = {name: string, checked: boolean};

  // Using Array.map instead of for loops for better performance from
  // Promise.all. Otherwise, it can take ~10 seconds to show the packages
  // if modifyName or selected take a long time.
  const choices = await Promise.all<SeparatorInstance | Promise<Choice>>(
    releaseUnits
      .map(releaseUnit => [
        new inquirer.Separator( // Separate release units with a line
          chalk.underline(releaseUnit.name)),
        ...releaseUnit.phases // Display the packages of a release unit.
          .map(phase => phase.packages
               .map(async pkg => {
                 const [name, checked] = await Promise.all([
                   modifyName(pkg), selected(pkg)]);
                 return {name, value: pkg, checked};
               }) // Promise<Choice>[] from one phase's packages
              ).flat() // Promise<Choice>[] from one release unit
      ]).flat() // (Separator | Promise<Choice>)[] for all release units
  );

  const choice = await inquirer.prompt({
    name: 'packages',
    type: 'checkbox',
    message,
    pageSize: 30,
    choices,
    loop: false,
  } as {name: 'packages'});

  return choice['packages'] as string[];
}

export function getVersion(packageJsonPath: string) {
  return JSON.parse(fs.readFileSync(packageJsonPath)
                    .toString('utf8')).version as string;
}

export function getLocalVersion(pkg: string) {
  return getVersion(path.join(pkg, 'package.json'));
}

export async function getNpmVersion(pkg: string, registry?: string,
                                    tag = 'latest') {
  const env: Record<string, string> = {};
  if (registry) {
    env['NPM_CONFIG_REGISTRY'] = registry;
  }
  return $async(`npm view @tensorflow/${pkg} dist-tags.${tag}`, env);
}

export function getTagFromVersion(version: string): string {
  if (version.includes('dev')) {
    return 'nightly';
  }else if (version.includes('rc')) {
    return 'next';
  }
  return 'latest';
}

export function memoize<I, O>(f: (arg: I) => Promise<O>): (arg: I) => Promise<O> {
  const map = new Map<I, Promise<O>>();
  return async (i: I) => {
    if (!map.has(i)) {
      map.set(i, f(i));
    }
    return map.get(i)!;
  }
}

export async function runVerdaccio(): Promise<() => void> {
  // Remove the verdaccio package store.
  // TODO(mattsoulanille): Move the verdaccio storage and config file here
  // once the nightly verdaccio tests are handled by this script.
  rimraf.sync(path.join(__dirname, '../e2e/scripts/storage'));

  // Start verdaccio. It must be started directly from its binary so that IPC
  // messaging works and verdaccio can tell node that it has started.
  // https://verdaccio.org/docs/verdaccio-programmatically/#using-fork-from-child_process-module
  const verdaccioBin = require.resolve('verdaccio/bin/verdaccio');
  const config = path.join(__dirname, '../e2e/scripts/verdaccio.yaml');
  const serverProcess = fork(verdaccioBin, [`--config=${config}`]);
  const ready = new Promise<void>((resolve, reject) => {
    const timeLimitMilliseconds = 30_000;
    console.log(`Waiting ${timeLimitMilliseconds / 1000} seconds for ` +
                'verdaccio to start....');
    const timeout = setTimeout(() => {
      serverProcess.kill();
      reject(`Verdaccio did not start in ${timeLimitMilliseconds} seconds.`);
    }, timeLimitMilliseconds);

    serverProcess.on('message', (msg: {verdaccio_started: boolean}) => {
      if (msg.verdaccio_started) {
        console.log(chalk.magenta.bold(
            `Verdaccio Started. Visit http://localhost:4873 to see packages.`));
        clearTimeout(timeout);
        resolve();
      }
    });
  });

  serverProcess.on('error', (err: unknown) => {
    throw new Error(`Verdaccio error: ${err}`);
  });

  const onUnexpectedDisconnect = (err: unknown) => {
    throw new Error(`Verdaccio process unexpectedly disconnected: ${err}`);
  };
  serverProcess.on('disconnect', onUnexpectedDisconnect);

  const killVerdaccio = () => {
    serverProcess.off('disconnect', onUnexpectedDisconnect);
    serverProcess.kill();
  };

  // Kill verdaccio when node exits.
  process.on('exit', killVerdaccio);

  await ready;
  return killVerdaccio;
}

/**
 * Check a package.json path for `link://` and `file://` dependencies.
 */
export function checkPublishable(packageJsonPath: string): void {
  const packageJson = JSON.parse(
    fs.readFileSync(packageJsonPath)
      .toString('utf8')) as {
        name?: string,
        private?: boolean,
        dependencies?: Record<string, string>,
      };

  if (!packageJson.name) {
    throw new Error(`${packageJsonPath} has no name.`);
  }
  const pkg = packageJson.name;
  if (packageJson.private) {
    throw new Error(`${pkg} is private.`);
  }

  if (packageJson.dependencies) {
    for (let [dep, depVersion] of Object.entries(packageJson.dependencies)) {
      const start = depVersion.slice(0,5);
      if (start === 'link:' || start === 'file:') {
        throw new Error(`${pkg} has a '${start}' dependency on ${dep}. `
                        + 'Refusing to publish.');
      }
    }
  }
}
