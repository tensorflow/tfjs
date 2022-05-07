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

// Register the CPU backend as a default backend for tests.
import '@tensorflow/tfjs-backend-cpu';
/**
 * This file is necessary so we register all test environments before we start
 * executing tests.
 */
import {setTestEnvs, setupTestFilters, TestFilter} from './jasmine_util';

// Set up a CPU test env as the default test env
setTestEnvs([{name: 'cpu', backendName: 'cpu', isDataSync: true}]);

const TEST_FILTERS: TestFilter[] = [
  {
    startsWith: 'fused conv2d ',
    excludes: [
      'basic in NCHW with',  // NCHW format with bias or prelu activation is not
                             // supported yet
      'im2row in NCHW with',
    ]
  },
];
const customInclude = (testName: string) => {
  //  NCHW format with bias or prelu activation is not supported yet.
  if (testName.startsWith('fused conv2d')) {
    if (testName.includes('basic in NCHW with') ||
        testName.includes('im2row in NCHW with')) {
      return false;
    }
  }
  return true;
};
setupTestFilters(TEST_FILTERS, customInclude);

// Import and run all the tests.
// This import, which registers all tests, must be a require because it must run
// after the test environment is set up.
// tslint:disable-next-line:no-require-imports
require('./tests');
