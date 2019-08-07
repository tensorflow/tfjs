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

setTestEnvs([{name: 'test-wasm', backendName: 'wasm', isDataSync: true}]);

const env = jasmine.getEnv();

/** Tests that have these substrings in their name will be included. */
const INCLUDE_LIST: string[] = ['add '];
/** Tests that have these substrings in their name will be excluded. */
const EXCLUDE_LIST: string[] = [
  'complex',                    // Complex numbers not yet implemented.
  'gradient',                   // Gradient is missing.
  'broadcast inner dim',        // Broadcast inner dim not yet supported.
  'broadcast each with 1 dim',  // Same as above.
  'broadcasting same rank Tensors different shape',  // Same as above.
  'upcasts when dtypes dont match',  // Uses the 'complex' dtype.
];

/**
 * Filter method that returns boolean, if a given test should run or be
 * ignored based on its name. The exclude list has priority over the include
 * list. Thus, if a test matches both the exclude and the include list, it
 * will be exluded.
 */
env.specFilter = spec => {
  const name = spec.getFullName();
  // Return false (skip the test) if the test is in the exclude list.
  for (let i = 0; i < EXCLUDE_LIST.length; ++i) {
    if (name.indexOf(EXCLUDE_LIST[i]) > -1) {
      return false;
    }
  }

  // Include all regular describe() tests.
  if (name.indexOf('test-wasm') < 0) {
    return true;
  }

  // Include all of the wasm specific tests.
  if (name.startsWith('wasm')) {
    return true;
  }

  // Include a describeWithFlags() test from tfjs-core only if the test is in
  // the include list.
  for (let i = 0; i < INCLUDE_LIST.length; ++i) {
    if (name.indexOf(INCLUDE_LIST[i]) > -1) {
      return true;
    }
  }
  // Otherwise ignore the test.
  return false;
};

// Import and run all the tests from core.
import '@tensorflow/tfjs-core/dist/tests';
