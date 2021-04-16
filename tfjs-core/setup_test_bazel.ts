/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 * This file is necessary so we register all test environments before we start
 * executing tests.
 */
import '@tensorflow/tfjs-core';
// Register the CPU backend as a default backend for tests.
import '@tensorflow/tfjs-backend-cpu';
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
// tslint:disable-next-line:no-imports-from-dist
import {setTestEnvs, setupTestFilters, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

const TEST_FILTERS: TestFilter[] = [];
const customInclude = (testName: string) => {
  if (testName.indexOf('tensor in worker') !== -1) {
    return false;
  }
  return true;
};
setupTestFilters(TEST_FILTERS, customInclude);

// Set up a CPU test env as the default test env
setTestEnvs([{name: 'cpu', backendName: 'cpu', isDataSync: true}]);

// Import and run all the tests.
// This import, which registers all tests, must be a require because it must run
// after the test environment is set up.
// tslint:disable-next-line:no-require-imports
require('@tensorflow/tfjs-core/dist/tests');
