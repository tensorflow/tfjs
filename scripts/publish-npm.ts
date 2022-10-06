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
import {RELEASE_UNITS, question, getReleaseBranch, checkoutReleaseBranch, ALPHA_RELEASE_UNIT, TFJS_RELEASE_UNIT, selectPackages, getLocalVersion, getNpmVersion, memoize, printReleaseUnit, publishable, runVerdaccio, ReleaseUnit} from './release-util';
import * as fs from 'fs';
import semverCompare from 'semver/functions/compare';
import * as child_process from 'child_process';

import {BAZEL_PACKAGES} from './bazel_packages';

const TMP_DIR = '/tmp/tfjs-publish';
const VERDACCIO_REGISTRY = 'http://127.0.0.1:4873/';

const parser = new argparse.ArgumentParser();
parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocal rather than the http protocol when cloning repos.'
});

parser.addArgument('--registry', {
  type: 'string',
  defaultValue: 'https://registry.npmjs.org/',
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

// async function sleep(ms: number) {
//   return new Promise(resolve => {
//     setTimeout(resolve, ms);
//   });
// }


// async function flaky<T>(f: () => Promise<T>, tries = 3, delay = 1000) {
//   let err: Error | undefined;
//   for (let i = 0; i < tries; i++) {
//     try {
//       return await f();
//     } catch (e) {
//       err = e;
//       console.warn(err);
//       await sleep(delay);
//     }
//   }
//   throw err;
// }

async function publish(pkg: string, registry: string, otp?: string,
                       build = true) {
  const registryEnv = {
    YARN_REGISTRY: registry,
    NPM_CONFIG_REGISTRY: registry,
  };
  function run(command: string) {
    const env = {...process.env, ...registryEnv}
    return child_process.execSync(command, {env}).toString('utf8');
  }

  const startDir = process.cwd();
  shell.cd(pkg);

  const res = publishable('./package.json');
  if (res instanceof Error) {
    throw res;
  }

  if (build) {
    console.log(chalk.magenta.bold(`~~~ Preparing package ${pkg}~~~`));
    console.log(chalk.magenta('~~~ Installing packages ~~~'));
    // tfjs-node-gpu needs to get some files from tfjs-node.
    if (pkg === 'tfjs-node-gpu') {
      run('yarn prep-gpu');
    }

    // Yarn above the other checks to make sure yarn doesn't change the lock
    // file.
    //console.log($('yarn', registryEnv));
    console.log(run('yarn'));
   // console.log(await flaky(() => $async('echo $YARN_REGISTRY && yarn', registryEnv)));

    console.log(chalk.magenta('~~~ Build npm ~~~'));

    if (pkg === 'tfjs-react-native' || BAZEL_PACKAGES.has(pkg)) {
      run('yarn build-npm');
    } else {
      run('yarn build-npm for-publish');
    }
  }

  // Used for nightly dev releases.
  const version = JSON.parse(fs.readFileSync('package.json')
                             .toString('utf8')).version as string;
  const nightly = version.includes('dev');

  const tag = nightly ? 'nightly' : 'latest';

  let otpFlag = '';
  if (otp) {
    otpFlag = `--otp=${otp} `;
  }

  console.log(
    chalk.magenta.bold(`~~~ Publishing ${pkg} to ${registry} with tag `
                       + `${tag} ~~~`));

  if (BAZEL_PACKAGES.has(pkg)) {
    let dashes = '-- --';
    if (pkg === 'tfjs-backend-webgpu') {
      // Special case for webgpu, which has an additional call to `yarn`
      // in publish-npm.
      dashes = '-- -- --';
    }
    run(`yarn publish-npm ${dashes} ${otpFlag} --tag={tag} --force`);
  } else {
    let login = '';
    if (registry === VERDACCIO_REGISTRY) {
      login = 'npx npm-cli-login -u user -p password -e user@example.com && ';
    }
    // run('npx npm-cli-login -u user -p password -e user@example.com'
    //     + ` -r ${registry} && npm publish ${otpFlag} --tag=${tag} --force`);

    // if (pkg.startsWith('tfjs-node')) {
    //   // Special case for tfjs-node* because it must publish the node addon
    //   // as well.
    //   run(`${login}yarn publish-npm ${otpFlag}`);
    // } else {
    run(`${login}npm publish ${otpFlag}`);
    // }
  }
  console.log(`Yay! Published ${pkg} to npm.`);

  shell.cd(startDir);
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
    const releaseUnitInt = +releaseUnitStr;
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

  const getNpmVersionMemoized = memoize(getNpmVersion);
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
        return pkg;
      }
    }
  });

  // Yarn in the top-level and in the directory.
  child_process.execSync('yarn');
  console.log();

  // Build and publish all packages to Verdaccio
  const verdaccio = runVerdaccio();
  //runVerdaccio;
  //const verdaccioRegistry = 'http://[::1]:4873/';
  //const verdaccioRegistry = 'http://localhost:4873/';

  // child_process.execSync('npx npm-cli-login -u user -p password -e user@example.com'
  //   + ` -r ${VERDACCIO_REGISTRY}`);
  for (const pkg of packages) {
    await publish(pkg, VERDACCIO_REGISTRY);
  }
  verdaccio.kill();

  if (args.dry) {
    console.log('Not publishing packages due to \'--dry\'');
  } else {
    // Publish all built packages to the selected registry
    let otp = '';
    if (!args.no_otp) {
      otp = await question(`Enter one-time password from your authenticator: `);
    }

    const promises = packages.map(pkg => publish(pkg, args.registry, otp, false));
    await Promise.all(promises);
    console.log(`Published packages to ${args.registry}`);
  }
  process.exit(0);
}

main();
