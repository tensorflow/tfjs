#!/usr/bin/env node
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as mkdirp from 'mkdirp';
import * as readline from 'readline';
import * as shell from 'shelljs';
import * as fs from 'fs';
import chalk from 'chalk';

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

interface Phase {
  repos: string[];
  deps?: string[];
  scripts?: string[];
  // Whether to leave the version alone. Defaults to false (change the version).
  leaveVersion?: boolean;
  optional?: boolean;
}

const PHASES: Phase[] = [
  {repos: ['tfjs-core']},
  {deps: ['tfjs-core'], repos: ['tfjs-layers', 'tfjs-converter', 'tfjs-data']},
  {
    deps: ['tfjs-core', 'tfjs-layers', 'tfjs-converter', 'tfjs-data'],
    repos: ['tfjs']
  },
  {deps: ['tfjs'], repos: ['tfjs-node']}, {repos: ['tfjs-vis'], optional: true},
  {
    deps: ['tfjs', 'tfjs-node', 'tfjs-vis'],
    repos: ['tfjs-website'],
    scripts: ['yarn build-prod'],
    leaveVersion: true
  }
];

const TMP_DIR = '/tmp/tfjs-release';

const RED_TERMINAL_COLOR = '\x1b[31m%s\x1b[0m';

function printPhase(phaseId: number) {
  const phase = PHASES[phaseId];
  console.log(chalk.green(`Phase ${phaseId}:`));
  if (phase.deps != null) {
    console.log(`  deps : ${phase.deps.join(', ')}`);
  }
  console.log(`  repos: ${phase.repos.join(', ')}`);
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
    console.log(RED_TERMINAL_COLOR, `Invalid phase: ${phaseStr}`);
    process.exit(1);
  }
  console.log(chalk.blue(`Using phase ${phaseInt}`));
  console.log();

  const phase = PHASES[phaseInt];
  const repos = PHASES[phaseInt].repos;

  for (let i = 0; i < repos.length; i++) {
    const repo = repos[i];

    mkdirp(TMP_DIR, (err) => {
      if (err) {
        console.log('Error creating temp dir', TMP_DIR);
        process.exit(1);
      }
    });
    $(`rm -f -r ${TMP_DIR}/${repo}/*`);

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

    if (phase.deps != null) {
      for (let j = 0; j < phase.deps.length; j++) {
        const dep = phase.deps[j];
        let version = null;
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
        const patchUpdateVersion = getPatchUpdateVersion(version);

        let depVersion = await question(
            `Updated version for ` +
            `${dep} (leave empty for ${patchUpdateVersion}): `);
        if (depVersion === '') {
          depVersion = patchUpdateVersion;
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

    console.log($(`git diff`));

    $(`git checkout -b b${newVersion}`);
    $(`git push -u origin b${newVersion}`);
    $(`git add .`);
    $(`git commit -a -m "Update ${repo} to ${newVersion}."`);
    $(`git push`);
    $(`hub pull-request --browse --message "Update ${repo} to ${newVersion}."`);
    console.log();
  }

  process.exit(0);
}
main();
