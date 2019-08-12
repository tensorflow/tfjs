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
 * repositories. The release process is split up into multiple phases. Each
 * phase will update the version of a package and the dependency versions and
 * send a pull request. Once the pull request is merged, you must publish the
 * packages manually from the individual repositories.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import * as mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as shell from 'shelljs';
import * as fs from 'fs';
import chalk from 'chalk';

interface Phase {
  // The list of repositories that will be updated with this change.
  repos: string[];
  // The list of dependencies that all of the repositories will update to.
  deps?: string[];
  // An ordered list of scripts to run after yarn is called and before the pull
  // request is sent out.
  scripts?: string[];
  // Whether to leave the version of the package alone. Defaults to false
  // (change the version).
  leaveVersion?: boolean;
  title?: string;
}

const CORE_PHASE: Phase = {
  repos: ['tfjs-core'],
  scripts: ['./scripts/make-version']
};

const LAYERS_CONVERTER_DATA_PHASE: Phase = {
  repos: ['tfjs-layers', 'tfjs-converter', 'tfjs-data'],
  deps: ['tfjs-core'],
  scripts: ['./scripts/make-version']
};

const UNION_PHASE: Phase = {
  repos: ['tfjs'],
  deps: ['tfjs-core', 'tfjs-layers', 'tfjs-converter', 'tfjs-data'],
  scripts: ['./scripts/make-version']
};

const NODE_PHASE: Phase = {
  repos: ['tfjs-node'],
  deps: ['tfjs'],
  scripts: ['./scripts/make-version']
};

const VIS_PHASE: Phase = {
  repos: ['tfjs-vis']
};

const WEBSITE_PHASE: Phase = {
  repos: ['tfjs-website'],
  deps: ['tfjs', 'tfjs-node', 'tfjs-vis'],
  scripts: ['yarn build-prod'],
  leaveVersion: true,
  title: 'Update website to latest dependencies.'
};

const PHASES: Phase[] = [
  CORE_PHASE, LAYERS_CONVERTER_DATA_PHASE, UNION_PHASE, NODE_PHASE, VIS_PHASE,
  WEBSITE_PHASE
];

const TMP_DIR = '/tmp/tfjs-release';

function printPhase(phaseId: number) {
  const phase = PHASES[phaseId];
  console.log(chalk.green(`Phase ${phaseId}:`));
  console.log(`  repos: ${chalk.blue(phase.repos.join(', '))}`);
  if (phase.deps != null) {
    console.log(`   deps: ${phase.deps.join(', ')}`);
  }
}

// Computes the default updated version (does a patch version update).
function getPatchUpdateVersion(version: string): string {
  const versionSplit = version.split('.');

  return [versionSplit[0], versionSplit[1], +versionSplit[2] + 1].join('.');
}

async function main() {
  PHASES.forEach((_, i) => printPhase(i));
  console.log();

  const phaseStr = await question('Which phase (leave empty for 0): ');
  const phaseInt = +phaseStr;
  if (phaseInt < 0 || phaseInt >= PHASES.length) {
    console.log(chalk.red(`Invalid phase: ${phaseStr}`));
    process.exit(1);
  }
  console.log(chalk.blue(`Using phase ${phaseInt}`));
  console.log();

  const phase = PHASES[phaseInt];
  const repos = PHASES[phaseInt].repos;
  const deps = PHASES[phaseInt].deps || [];

  for (let i = 0; i < repos.length; i++) {
    const repo = repos[i];

    mkdirp(TMP_DIR, (err) => {
      if (err) {
        console.log('Error creating temp dir', TMP_DIR);
        process.exit(1);
      }
    });
    $(`rm -f -r ${TMP_DIR}/${repo}/*`);
    $(`rm -f -r ${TMP_DIR}/${repo}`);

    const depsLatestVersion: string[] =
        deps.map(dep => $(`npm view @tensorflow/${dep} dist-tags.latest`));

    const dir = `${TMP_DIR}/${repo}`;
    $(`mkdir ${dir}`);
    $(`git clone https://github.com/tensorflow/${repo} ${dir} --depth=1`);

    shell.cd(dir);

    // Update the version.
    let pkg = `${fs.readFileSync(`${dir}/package.json`)}`;
    const parsedPkg = JSON.parse(`${pkg}`);

    console.log(chalk.magenta.bold(
        `~~~ Processing ${repo} (${parsedPkg.version}) ~~~`));

    const patchUpdateVersion = getPatchUpdateVersion(parsedPkg.version);
    let newVersion = parsedPkg.version;
    if (!phase.leaveVersion) {
      newVersion = await question(
          `New version (leave empty for ${patchUpdateVersion}): `);
      if (newVersion === '') {
        newVersion = patchUpdateVersion;
      }
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

    fs.writeFileSync(`${dir}/package.json`, pkg);
    $(`yarn`);
    if (phase.scripts != null) {
      phase.scripts.forEach(script => $(script));
    }

    $(`git checkout -b b${newVersion}`);
    $(`git push -u origin b${newVersion}`);
    $(`git add .`);
    $(`git commit -a -m "Update ${repo} to ${newVersion}."`);
    $(`git push`);
    const title =
        phase.title ? phase.title : `Update ${repo} to ${newVersion}.`;
    $(`hub pull-request --browse --message "${title}" --labels INTERNAL`);
    console.log();
  }

  console.log(
      `Done. FYI, this script does not publish to NPM. ` +
      `Please publish by running ./scripts/publish-npm.sh ` +
      `from each repo after you merge the PR.`);

  process.exit(0);
}

/**
 * A wrapper around shell.exec for readability.
 * @param cmd The bash command to execute.
 * @returns stdout returned by the executed bash script.
 */
function $(cmd: string) {
  const result = shell.exec(cmd, {silent: true});
  if (result.code > 0) {
    console.log('$', cmd);
    console.log(result.stderr);
    process.exit(1);
  }
  return result.stdout.trim();
}

const rl =
    readline.createInterface({input: process.stdin, output: process.stdout});

async function question(questionStr: string): Promise<string> {
  console.log(chalk.bold(questionStr));
  return new Promise<string>(
      resolve => rl.question('> ', response => resolve(response)));
}

main();
