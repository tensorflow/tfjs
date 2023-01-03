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

/*
 * This script publish to npm for all the TensorFlow.js packages. Before you run
 * this script, run `yarn release` and commit the PRs.
 * Then run this script as `yarn publish-npm`.
 */

import * as argparse from 'argparse';
import chalk from 'chalk';
import * as shell from 'shelljs';
import { RELEASE_UNITS, question, $, getReleaseBranch, checkoutReleaseBranch, ALPHA_RELEASE_UNIT, TFJS_RELEASE_UNIT, selectPackages, getLocalVersion, getNpmVersion, memoize, printReleaseUnit, publishable, runVerdaccio, ReleaseUnit, getVersion, getTagFromVersion } from './release-util';
import semverCompare from 'semver/functions/compare';
import * as child_process from 'child_process';

import {BAZEL_PACKAGES} from './bazel_packages';

const TMP_DIR = '/tmp/tfjs-publish';
const VERDACCIO_REGISTRY = 'http://127.0.0.1:4873';
const NPM_REGISTRY = 'https://registry.npmjs.org/';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

parser.addArgument('--registry', {
  type: 'string',
  defaultValue: NPM_REGISTRY,
  help: 'Which registry to install packages from and publish to.',
});

parser.addArgument('--no-otp', {
  action: 'storeTrue',
  help: 'Do not use an OTP when publishing to the registry.',
});

parser.addArgument(['--release-this-branch', '--release-current-branch'], {
  action: 'storeTrue',
  help: 'Release the current branch instead of checking out a new one.',
});

parser.addArgument(['--dry'], {
  action: 'storeTrue',
  help: 'Dry run. Stage all packages in verdaccio but do not publish them to '
      + 'the registry.',
});

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function publish(pkg: string, registry: string, otp?: string,
                       build = true) {
  const registryEnv = {
    YARN_REGISTRY: registry,
    NPM_CONFIG_REGISTRY: registry,
    NPM_REGISTRY: registry, // For npx npm-cli-login
  };

  function run(command: string) {
    return $(command, registryEnv);
  }

  function yarn(args: string) {
    return run(`yarn --registry '${registry}' ${args}`);
  }

  const startDir = process.cwd();
  try {
    // Use a try block so cwd can be restored in 'finally' if an error occurs.
    shell.cd(pkg);

    const res = publishable('./package.json');
    if (res instanceof Error) {
      throw res;
    }

    if (build && !BAZEL_PACKAGES.has(pkg)) {
      console.log(chalk.magenta.bold(`~~~ Preparing package ${pkg}~~~`));
      console.log(chalk.magenta('~~~ Installing packages ~~~'));
      // Without a delay, this sometimes has issues downloading dependencies.
      await delay(5_000);

      // tfjs-node-gpu needs to get some files from tfjs-node.
      if (pkg === 'tfjs-node-gpu') {
        yarn('prep-gpu');
      }

      // Yarn above the other checks to make sure yarn doesn't change the lock
      // file.
      console.log(run(`yarn --registry '${registry}'`));
      console.log(chalk.magenta('~~~ Build npm ~~~'));

      if (pkg === 'tfjs-react-native') {
        yarn('build-npm');
      } else {
        yarn('build-npm for-publish');
      }
    }

    // Used for nightly dev releases.
    const version = getVersion('package.json');
    const tag = getTagFromVersion(version);

    let otpFlag = '';
    if (otp) {
      otpFlag = `--otp=${otp} `;
    }

    console.log(
      chalk.magenta.bold(`~~~ Publishing ${pkg} to ${registry} with tag `
        + `${tag} ~~~`));

    let login = '';
    if (registry === VERDACCIO_REGISTRY) {
      // If publishing to verdaccio, we must log in before every command.
      login = 'npx npm-cli-login -u user -p password -e user@example.com && ';
    }

    if (BAZEL_PACKAGES.has(pkg)) {
      let dashes = '-- --';
      if (pkg === 'tfjs-backend-webgpu') {
        // Special case for webgpu, which has an additional call to `yarn`
        // in publish-npm.
        dashes = '-- -- --';
      }
      run(`${login}yarn --registry '${registry}' publish-npm ${dashes} ${otpFlag} --tag=${tag} --force`);
    } else {
      if (registry === NPM_REGISTRY && pkg.startsWith('tfjs-node')) {
        // Special case for tfjs-node(-gpu), which must upload the node addon
        // to GCP as well. Only do this when publishing to NPM.
        $('yarn build-and-upload-addon publish');
      }

      // Publish the package to the registry.
      run(`${login}npm --registry '${registry}' publish ${otpFlag}`);
    }
    console.log(`Yay! Published ${pkg} to ${registry}.`);

  } finally {
    shell.cd(startDir);
  }
}

async function main() {
  const args = parser.parseArgs();

  let releaseUnits: ReleaseUnit[];
  if (args.release_this_branch) {
    console.log('Releasing current branch');
    releaseUnits = RELEASE_UNITS;
  } else {
    RELEASE_UNITS.forEach(printReleaseUnit);
    console.log();

    const releaseUnitStr =
      await question('Which release unit (leave empty for 0): ');
    const releaseUnitInt = Number(releaseUnitStr);
    if (releaseUnitInt < 0 || releaseUnitInt >= RELEASE_UNITS.length) {
      console.log(chalk.red(`Invalid release unit: ${releaseUnitStr}`));
      process.exit(1);
    }
    console.log(chalk.blue(`Using release unit ${releaseUnitInt}`));
    console.log();

    const releaseUnit = RELEASE_UNITS[releaseUnitInt];
    const {name, } = releaseUnit;

    let releaseBranch: string;
    if (releaseUnit === ALPHA_RELEASE_UNIT) {
      // Alpha release unit is published with the tfjs release unit.
      releaseBranch = await getReleaseBranch(TFJS_RELEASE_UNIT.name);
    } else {
      releaseBranch = await getReleaseBranch(name);
    }
    console.log();

    releaseUnits = [releaseUnit];
    checkoutReleaseBranch(releaseBranch, args.git_protocol, TMP_DIR);
    shell.cd(TMP_DIR);
  }

  const getNpmVersionMemoized = memoize((pkg: string) => {
    const version = getLocalVersion(pkg);
    const tag = getTagFromVersion(version);
    return getNpmVersion(pkg, args.registry, tag);
  });

  const packages = await selectPackages({
    message: 'Select packages to publish',
    releaseUnits,
    async selected(pkg) {
      // Automatically select local packages with version numbers greater than
      // npm.
      try {
        const localVersion = getLocalVersion(pkg);
        const npmVersion = await getNpmVersionMemoized(pkg);
        const localIsNewer = semverCompare(localVersion, npmVersion) > 0;
        return localVersion !== '0.0.0' && localIsNewer;
      } catch (e) {
        console.warn(e);
        return false;
      }
    },
    async modifyName(pkg) {
      // Add the local and remote versions to the printed name.
      try {
        const localVersion = getLocalVersion(pkg);
        const npmVersion = await getNpmVersionMemoized(pkg);
        const localIsNewer = semverCompare(localVersion, npmVersion) > 0;
        const pkgWithVersion =
          `${pkg.padEnd(20)} (${npmVersion} → ${localVersion})`;
        if (localIsNewer) {
          return chalk.bold(pkgWithVersion);
        } else {
          return pkgWithVersion;
        }
      } catch (e) {
        console.warn(e);
        return pkg;
      }
    }
  });

  // Yarn in the top-level and in the directory.
  child_process.execSync('yarn');
  console.log();

  // Build and publish all packages to Verdaccio
  const verdaccio = runVerdaccio();
  try {
    for (const pkg of packages) {
      await publish(pkg, VERDACCIO_REGISTRY);
    }
  } finally {
    verdaccio.kill();
  }

  if (args.dry) {
    console.log('Not publishing packages due to \'--dry\'');
  } else {
    // Publish all built packages to the selected registry
    let otp = '';
    if (!args.no_otp) {
      otp = await question(`Enter one-time password from your authenticator: `);
    }
    console.log(`Publishing packages to ${args.registry}`);

    const toPublish = [...packages];
    while (toPublish.length > 0) {
      let pkg = toPublish[0];
      if (args.no_otp) {
        await publish(pkg, args.registry, '', false);
        toPublish.shift(); // Remove the published package from 'toPublish'.
        continue;
      }

      try {
        await publish(pkg, args.registry, otp, false)
        toPublish.shift(); // Remove the published package from 'toPublish'.
      } catch (err) {
        if ((err as Error).message.includes('code EOTP')) {
          // Try again with a new otp
          otp = await question(`OTP ${otp} failed. Enter a new one-time `
                               + `password from your authenticator: `);
          continue; // Don't shift the package since it failed to publish.
        }
        throw err;
      }
    }

    console.log(`Published packages to ${args.registry}`);
  }
  process.exit(0);
}

main();
