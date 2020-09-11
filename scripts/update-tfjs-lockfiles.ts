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

import * as argparse from 'argparse';
import * as shell from 'shelljs';
import {exec} from 'child_process';

import {$, TFJS_RELEASE_UNIT, prepareReleaseBuild, getReleaseBranch, checkoutReleaseBranch} from './release-util';

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

  // ========== Push to release branch. ========================================
  const message = `Update release branch ${releaseBranch} lock files.`;

  $(`git add .`);
  $(`git commit -a -m "${message}"`);
  $(`git push`);

  // ========== Tag version. ========================================
  console.log('~~~ Tag version ~~~');
  shell.cd('..');

  // The releaseBranch format is tfjs_x.x.x, we only need the version part.
  const version = releaseBranch.split('_')[1];
  const tag = `tfjs-v${version}`;
  exec(`git tag ${tag} && git push --tags`, (err) => {
    if (err) {
      throw new Error(`Could not git tag with ${tag}: ${err.message}.`);
    }
    console.log(`Successfully tagged with ${tag}.`);
  });

  console.log('Done.');

  process.exit(0);
}

main();
