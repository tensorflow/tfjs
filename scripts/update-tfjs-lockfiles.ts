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
 * This script update yarn.lock after tfjs packages are published.
 *
 * This script requires hub to be installed: https://hub.github.com/
 */

import chalk from 'chalk';
import * as argparse from 'argparse';
import * as shell from 'shelljs';

import {$, TFJS_RELEASE_UNIT, prepareReleaseBuild, getReleaseBranch, checkoutReleaseBranch, createPR} from './release-util';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
});

const TMP_DIR = '/tmp/tfjs-publish-after';

async function main() {
  const args = parser.parseArgs();

  // ========== Get release branch. ============================================
  // Infer release branch name.
  let releaseBranch = await getReleaseBranch('tfjs');
  console.log();

  // ========== Checkout release branch. =======================================
  checkoutReleaseBranch(releaseBranch, args.git_protocol, TMP_DIR);

  shell.cd(TMP_DIR);

  // ========== Delete a possible prior lockfiles branch from a failed run =====
  const lockfilesBranch = `${releaseBranch}_lockfiles`;
  console.log(chalk.magenta.bold(
      `~~~ Creating new lockfiles branch ${lockfilesBranch} ~~~`));

  // Delete possible branch from a prior execution of this script
  const branchCmd = `git branch -D ${lockfilesBranch}`;
  const result = shell.exec(branchCmd, {silent: true});
  const okErrCode = `error: branch '${lockfilesBranch}' not found.`;
  if (result.code > 0 && result.stderr.trim() !== okErrCode) {
    console.log('$', branchCmd);
    console.log(result.stderr);
    process.exit(1);
  }

  // ========== Run yarn to update yarn.lock file for each package. ============
  // Yarn in the top-level.
  $('yarn');

  // run yarn for every tfjs package
  const phases = TFJS_RELEASE_UNIT.phases;

  for (let i = 0; i < phases.length; i++) {
    const phase = phases[i];
    const packages = phase.packages;

    for (let i = 0; i < packages.length; i++) {
      const packageName = packages[i];
      shell.cd(packageName);

      prepareReleaseBuild(phase, packageName);

      shell.cd('..');
    }
  }

  // ========== Send a PR to the release branch =====================
  const message = `Update lockfiles branch ${lockfilesBranch} lock files.`;
  createPR(lockfilesBranch, releaseBranch, message);

  console.log('Done.');

  process.exit(0);
}

main();
