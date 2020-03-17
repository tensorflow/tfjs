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

interface Phase {
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

interface ReleaseUnit {
  // A human-readable name. Used for generating release branch.
  name: string;
  // The phases in this release unit.
  phases: Phase[];
  // The repository *only if it is not the same as tfjs*.
  repo?: string;
}

const CORE_PHASE: Phase = {
  packages: ['tfjs-core']
};

const LAYERS_CONVERTER_PHASE: Phase = {
  packages: ['tfjs-layers', 'tfjs-converter'],
  deps: ['tfjs-core']
};

const DATA_PHASE: Phase = {
  packages: ['tfjs-data'],
  deps: ['tfjs-core', 'tfjs-layers']
}

const UNION_PHASE: Phase = {
  packages: ['tfjs'],
  deps: ['tfjs-core', 'tfjs-layers', 'tfjs-converter', 'tfjs-data']
};

const NODE_PHASE: Phase = {
  packages: ['tfjs-node', 'tfjs-node-gpu'],
  deps: ['tfjs', 'tfjs-core'],
  scripts: {'tfjs-node-gpu': {'before-yarn': ['yarn prep-gpu']}}
};

const WASM_PHASE: Phase = {
  packages: ['tfjs-backend-wasm'],
  deps: ['tfjs-core']
};

const VIS_PHASE: Phase = {
  packages: ['tfjs-vis']
};

const REACT_NATIVE_PHASE: Phase = {
  packages: ['tfjs-react-native']
};

const WEBSITE_PHASE: Phase = {
  packages: ['tfjs-website'],
  deps: ['tfjs', 'tfjs-node', 'tfjs-vis', 'tfjs-react-native'],
  scripts: {'tfjs-website': {'after-yarn': ['yarn build-prod']}},
  leaveVersion: true,
  title: 'Update website to latest dependencies.'
};

const TFJS_RELEASE_UNIT: ReleaseUnit = {
  name: 'tfjs',
  phases: [
    CORE_PHASE, LAYERS_CONVERTER_PHASE, DATA_PHASE, UNION_PHASE, NODE_PHASE,
    WASM_PHASE
  ]
};

const VIS_RELEASE_UNIT: ReleaseUnit = {
  name: 'vis',
  phases: [VIS_PHASE]
};

const REACT_NATIVE_RELEASE_UNIT: ReleaseUnit = {
  name: 'react-native',
  phases: [REACT_NATIVE_PHASE]
};

const WEBSITE_RELEASE_UNIT: ReleaseUnit = {
  name: 'website',
  phases: [WEBSITE_PHASE],
  repo: 'tfjs-website'
};

const RELEASE_UNITS: ReleaseUnit[] = [
  TFJS_RELEASE_UNIT, VIS_RELEASE_UNIT, REACT_NATIVE_RELEASE_UNIT, WEBSITE_RELEASE_UNIT
];

const TMP_DIR = '/tmp/tfjs-release';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

function printReleaseUnit(id: number) {
  const releaseUnit = RELEASE_UNITS[id];
  console.log(chalk.green(`Release unit ${id}:`));
  console.log(` packages: ${chalk.blue(releaseUnit.phases.map(
    phase => phase.packages.join(', ')).join(', '))}`);
}

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

function getNextPatchVersion(phases: Phase, currentPhaseNumber: number): string {
  const latestVersion = $(`npm view @tensorflow/${phases[0].packages[0]} dist-tags.latest`);

  // Assumption: Phase0 is always published before the rest of the
  // phases.
  return currentPhaseNumber === 0 ? getPatchUpdateVersion(latestVersion) :
    latestVersion;
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

  const dir = `${TMP_DIR}/${repo == null ? `tfjs` : repo}`;
  mkdirp(TMP_DIR, err => {
    if (err) {
      console.log('Error creating temp dir', TMP_DIR);
      process.exit(1);
    }
  });
  $(`rm -f -r ${dir}/*`);
  $(`rm -f -r ${dir}`);
  $(`mkdir ${dir}`);

  const urlBase = args.git_protocol ? 'git@github.com:' : 'https://github.com/';
  let releaseBranch = '';

  if (repo != null) {
    // Publishing website, another repo.
    $(`git clone ${urlBase}tensorflow/${repo} ${dir} --depth=1`);
    shell.cd(dir);
  } else {
    // Publishing packages in tfjs.
    if (phaseInt !== 0) {
      // Phase0 should be published and release branch should have been created.
      const latestVersion =
        $(`npm view @tensorflow/${phases[0].packages[0]} dist-tags.latest`);
      releaseBranch = `${name}_${latestVersion}`;
      $(`git clone -b ${releaseBranch} ${urlBase}tensorflow/tfjs ${
        dir} --depth=1`);
      shell.cd(dir);
    } else {
      // Phase0 needs user input of the release version to create release branch.
      $(`git clone ${urlBase}tensorflow/tfjs ${dir} --depth=1`);
      shell.cd(dir);
    }
  }

  const newVersions = [];
  for (let i = 0; i < packages.length; i++) {
    const packageName = packages[i];
    if (repo == null) {
      shell.cd(packageName);
    }

    const depsLatestVersion: string[] =
      deps.map(dep => $(`npm view @tensorflow/${dep} dist-tags.latest`));

    // Update the version.
    const packageJsonPath = repo == null ?
      `${dir}/${packageName}/package.json` :
      `${dir}/package.json`
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
      console.log(`~~~ Creating new release branch ${releaseBranch}~~~`);
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
    if (repo == null) {
      shell.cd('..');
    }
    if (!phase.leaveVersion) {
      $(`./scripts/make-version.js ${packageName}`);
    }
    newVersions.push(newVersion);
  }

  const packageNames = packages.join(', ');
  const versionNames = newVersions.join(', ');
  const branchName = `b${newVersions.join('-')}_phase${phaseInt}`;
  $(`git checkout -b ${branchName}`);
  $(`git push -u origin ${branchName}`);
  $(`git add .`);
  $(`git commit -a -m "Update ${packageNames} to ${versionNames}."`);
  $(`git push`);
  const title =
    phase.title ? phase.title : `Update ${packageNames} to ${versionNames}.`;
  $(`hub pull-request -b ${releaseBranch} -m "${title}" -l INTERNAL -o`);
  console.log();

  console.log(
    `Done. FYI, this script does not publish to NPM. ` +
    `Please publish by running ./scripts/publish-npm.sh ` +
    `from each repo after you merge the PR.` +
    `Please remeber to update the website once you have released ` +
    'a new package version');

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
