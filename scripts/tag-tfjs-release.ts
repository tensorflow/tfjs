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
 * This script tags a release after lockfiles are updated
 */

import * as argparse from 'argparse';
import * as shell from 'shelljs';
import {$, checkoutReleaseBranch, getReleaseBranch} from './release-util';

const parser = new argparse.ArgumentParser();

parser.addArgument('--git-protocol', {
  action: 'storeTrue',
  help: 'Use the git protocol rather than the http protocol when cloning repos.'
});

async function main() {
  const args = parser.parseArgs();
  // ========== Get release branch. ============================================
  const releaseBranch = await getReleaseBranch('tfjs');

  const TMP_DIR = '/tmp/tfjs-tag';

  // ========== Checkout release branch. =======================================
  checkoutReleaseBranch(releaseBranch, args.git_protocol, TMP_DIR);

  shell.cd(TMP_DIR);

  // ========== Tag the release. ===============================================
  const version = releaseBranch.split('_')[1];
  const tag = `tfjs-v${version}`;
  console.log(`Tagging with ${tag}`);
  $(`git tag ${tag} && git push --tags`);
  console.log('Done.');

  // So the script doesn't just hang.
  process.exit(0);
}

main();
