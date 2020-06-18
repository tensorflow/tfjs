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
 * this script, run `yarn release-all` and commit the PR.
 * Then run this script as `yarn publish-all`.
 */

import * as shell from 'shelljs';
import {$, TFJS_RELEASE_UNIT} from './release-util';

async function main() {
  const {phases} = TFJS_RELEASE_UNIT;

  phases.forEach((phase) => {
    const packages = phase.packages;

    for (let i = 0; i < packages.length; i++) {
      const pkg = packages[i];
      shell.cd(pkg);

      console.log(`~~~ Checking package ${pkg}~~~`);
      const status = $('git status --porcelain');

      if (status != '') {
        console.log('Your git status is not clean. Aborting.');
        process.exit(1);
      }

      console.log('~~~ Installing packages ~~~');

      // tfjs-node-gpu needs to get some files from tfjs-node.
      if (pkg === 'tfjs-node-gpu') {
        $('yarn prep-gpu');
      }

      // tfjs-backend-wasm needs emsdk to build.
      if (pkg === 'tfjs-backend-wasm') {
        shell.cd('..');
        $('git clone https://github.com/emscripten-core/emsdk.git');
        shell.cd('./emsdk');
        $('./emsdk install 1.39.15');
        $('./emsdk activate 1.39.15');
        shell.cd('..');
        shell.cd(pkg);
      }

      // Yarn above the other checks to make sure yarn doesn't change the lock
      // file.
      $('yarn');

      console.log('~~~ Build npm ~~~');

      if (pkg === 'tfjs-backend-wasm') {
        // tfjs-backend-wasm needs emsdk env variables to build.
        $('source ../emsdk/emsdk_env.sh && yarn build-npm for-publish');
      } else {
        $('yarn build-npm for-publish');
      }

      console.log(`~~~ Publishing ${pkg} to npm ~~~`);
      shell.cd(pkg);
      $(`npm publish}`);
      console.log(`Yay! Published ${pkg} to npm.`);

      shell.cd('..');
      console.log();
    }
  });

  process.exit(0);
}

main();
