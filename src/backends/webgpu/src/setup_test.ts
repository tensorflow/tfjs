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

import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

setTestEnvs([{name: 'test-webgpu', backendName: 'webgpu', flags: {}}]);

const env = jasmine.getEnv();

/** Tests that have these substrings in their name will be included. */
const INCLUDE_LIST: string[] = [
  // TODO(smilkov): Enable this when we use async .data() in core tests.
  // 'matmul'
];
/** Tests that have these substrings in their name will be excluded. */
const EXCLUDE_LIST: string[] = [];

/**
 * Filter method that returns boolean, if a given test should run or be
 * ignored based on its name. The exclude list has priority over the include
 * list. Thus, if a test matches both the exclude and the include list, it will
 * be exluded.
 */
env.specFilter = spec => {
  const name = spec.getFullName();
  // Return false (skip the test) if the test is in the exclude list.
  for (let i = 0; i < EXCLUDE_LIST.length; ++i) {
    if (name.indexOf(EXCLUDE_LIST[i]) > -1) {
      return false;
    }
  }

  // Include all tests inside a regular describe().
  if (name.indexOf('test-webgpu') === -1) {
    return true;
  }

  // Include a test inside describeWithFlags() only if the test was in the
  // include list.
  for (let i = 0; i < INCLUDE_LIST.length; ++i) {
    if (name.indexOf(INCLUDE_LIST[i]) > -1) {
      return true;
    }
  }
  // Otherwise ignore the test.
  return false;
};

// Import and run all the tests from core.
async function runCoreTests() {
  // TODO(smilkov): Enable when we publish new core.
  // await tf.ready();
  // require('@tensorflow/tfjs-core/dist/tests');
}
runCoreTests();
