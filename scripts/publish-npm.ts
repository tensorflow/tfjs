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
import { RELEASE_UNITS, question, $, getReleaseBranch, checkoutReleaseBranch, ALPHA_RELEASE_UNIT, TFJS_RELEASE_UNIT, selectPackages, getLocalVersion, getNpmVersion, memoize, printReleaseUnit, checkPublishable, runVerdaccio, ReleaseUnit, getVersion, getTagFromVersion, filterPackages, ALL_PACKAGES, WEBSITE_RELEASE_UNIT, getPackages } from './release-util';
import semverCompare from 'semver/functions/compare';
import * as child_process from 'child_process';

import {BAZEL_PACKAGES} from './bazel_packages';

const TMP_DIR = '/tmp/tfjs-publish';
const VERDACCIO_REGISTRY = 'http://127.0.0.1:4873';
const NPM_REGISTRY = 'https://registry.npmjs.org/';

// This script can not publish the tfjs website
const PUBLISHABLE_RELEASE_UNITS = RELEASE_UNITS.filter(r => r !== WEBSITE_RELEASE_UNIT);

async function retry<T>(f: () => T, tries = 3, sleep=5_000): Promise<T> {
  let lastError;
  for (let i = 0; i < tries; i++) {
    try {
      return f();
    } catch (e) {
      lastError = e;
      console.warn(e);
      if (i + 1 < tries) {
        // Only delay if the loop will run again.
        await delay(sleep);
      }
    }
  }
  throw lastError;
}

/**
 * For sets `a` and `b`, compute the set difference `a \ b`
 *
 * The set difference of `a` and `b`, denoted `a \ b`, is the set containing all
 * elements of `a` that are not in `b`
 *
 * @param a The set to subtract from
 * @param b The set to remove from `a` when creating the output set
 */
function setDifference<T>(a: Set<T>, b: Set<T>): Set<T> {
  const difference = new Set<T>();
  for (const val of a) {
    if (!b.has(val)) {
      difference.add(val);
    }
  }
  return difference;
}

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
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

parser.addArgument(['--auto-publish-local-newer'], {
  action: 'storeTrue',
  help: 'Automatically publish local packages that have newer versions than'
      + ' the packages in the registry',
});

parser.addArgument(['--ci'], {
  action: 'storeTrue',
  help: 'Enable CI bazel flags for faster compilation and don\'t ask for user '
      + 'input before closing the verdaccio server once tests are done. '
      + 'Has no effect on results.',
});

parser.addArgument(['packages'], {
  type: 'string',
  nargs: '*',
  help: 'Packages to publish. Leave empty to select interactively',
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

    checkPublishable('./package.json');

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
      await retry(() =>
          console.log(run(`yarn --registry '${registry}'`)));
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
      await retry(() =>
          run(`${login}yarn --registry '${registry}' publish-npm ${dashes} ${otpFlag} --tag=${tag} --force`));
    } else {
      // Special case for tfjs-node(-gpu), which must upload the node addon
      // to GCP as well. Only do this when publishing to NPM.
      if (registry === NPM_REGISTRY && pkg.startsWith('tfjs-node')) {
        $('yarn build-and-upload-addon publish');
      }

      // Publish the package to the registry.
      await retry(() =>
          run(`${login}npm --registry '${registry}' publish --tag=${tag} ${otpFlag}`));
    }
    console.log(`Published ${pkg} to ${registry}.`);

  } finally {
    shell.cd(startDir);
  }
}

async function main() {
  const args = parser.parseArgs();

  const killVerdaccio = await runVerdaccio();

  let releaseUnits: ReleaseUnit[];
  if (args.release_this_branch) {
    console.log('Releasing current branch');
    releaseUnits = PUBLISHABLE_RELEASE_UNITS;
  } else {
    PUBLISHABLE_RELEASE_UNITS.forEach(printReleaseUnit);
    console.log();

    const releaseUnitStr =
      await question('Which release unit (leave empty for 0): ');
    const releaseUnitInt = Number(releaseUnitStr);
    if (releaseUnitInt < 0 || releaseUnitInt >= PUBLISHABLE_RELEASE_UNITS.length) {
      console.log(chalk.red(`Invalid release unit: ${releaseUnitStr}`));
      process.exit(1);
    }
    console.log(chalk.blue(`Using release unit ${releaseUnitInt}`));
    console.log();

    const releaseUnit = PUBLISHABLE_RELEASE_UNITS[releaseUnitInt];
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

  async function getVersions(pkg: string) {
    const localVersion = getLocalVersion(pkg);
    const npmVersion = await getNpmVersionMemoized(pkg);
    let localIsNewer = true;
    if (npmVersion !== '') {
      // Unpublished tags return '' for their version.
      localIsNewer = semverCompare(localVersion, npmVersion) > 0;
    }
    return {localVersion, npmVersion, localIsNewer};
  }

  async function packageSelected(pkg: string) {
      // Automatically select local packages with version numbers greater than
      // npm.
      try {
        const {localVersion, localIsNewer} = await getVersions(pkg);
        return localVersion !== '0.0.0' && localIsNewer;
      } catch (e) {
        console.warn(e);
        return false;
      }
  }

  // Get the list of packages to build and publish.
  // There are three ways packages can be selected.
  // 1. By passing them as CLI arguments in `packages`.
  // 2. Automatically based on the versions on npm.
  // 3. Interactively on the command line.
  let packages: string[];
  if (args.packages.length > 0) {
    // Get packages to publish from the 'packages' arg
    // Filter from the set of all packages to make sure they end up
    // in topological order.
    const allPackages = getPackages(PUBLISHABLE_RELEASE_UNITS);
    const requestedPackages = new Set(args.packages);
    packages = allPackages.filter(pkg => requestedPackages.has(pkg));

    // Check if there are any unsupported packages requested by the user
    const unsupportedPackages = setDifference(requestedPackages,
                                              new Set(packages));
    if (unsupportedPackages.size > 0) {
      throw new Error(`Can not publish ${[...unsupportedPackages]}. `
              + `Supported packages are:\n${[...ALL_PACKAGES].join('\n')}`);
    }
  } else if (args.auto_publish_local_newer) {
    // Automatically select packages based on npm versions
    packages = await filterPackages(packageSelected, PUBLISHABLE_RELEASE_UNITS);
    console.log(`Publishing ${packages}`);
  } else {
    // Select packages interactively
    packages = await selectPackages({
      message: 'Select packages to publish',
      releaseUnits,
      selected: packageSelected,
      async modifyName(pkg) {
        // Add the local and remote versions to the printed name.
        try {
          const {localVersion, npmVersion, localIsNewer} = await getVersions(pkg);
          const pkgWithVersion =
            `${pkg.padEnd(20)} (${npmVersion ?? 'unpublished'} → ${localVersion})`;
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
  }

  // Yarn in the top-level to download Bazel
  $('yarn');
  console.log();

  // Pre-build all the bazel packages in a single bazel command for better
  // efficiency.
  const bazelTargets = packages.filter(pkg => BAZEL_PACKAGES.has(pkg))
    .map(name => `//${name}:${name}_pkg`);

  const bazelArgs = ['bazel', 'build']
  if (args.ci) {
    bazelArgs.push('--config=ci');
  }
  // Use child_process.spawnSync to show bazel build progress.
  const result = child_process.spawnSync('yarn',
                                         [...bazelArgs, ...bazelTargets],
                                         {stdio:'inherit'});
  if (result.status !== 0) {
    throw new Error(`Bazel process failed with exit code ${result.status}`);
  }

  // Build and publish all packages to a local Verdaccio repo for staging.
  console.log(
    chalk.magenta.bold('~~~ Staging packages locally in Verdaccio ~~~'));

  try {
    for (const pkg of packages) {
      await publish(pkg, VERDACCIO_REGISTRY);
    }
  } catch (e) {
    // Make sure to kill the verdaccio server before exiting even if publish
    // throws an error. Otherwise, it blocks the port for the next run.
    killVerdaccio();
    throw e;
  }

  if (args.dry) {
    console.log('Not publishing packages due to \'--dry\'');
    if (!args.ci) {
      await question('Press enter to quit verdaccio.');
    }
    killVerdaccio();
  } else {
    // Publish all built packages to the selected registry
    let otp = '';
    if (!args.no_otp) {
      otp = await question(`Enter one-time password from your authenticator: `);
    }
    console.log(`Publishing packages to ${args.registry}`);

    killVerdaccio();

    const toPublish = [...packages];
    while (toPublish.length > 0) {
      // Using a while loop instead of .map since a stale OTP will require
      // a retry.
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
