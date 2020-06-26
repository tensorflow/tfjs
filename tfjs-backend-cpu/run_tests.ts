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

// Register the backend.
import './src/index';
// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs, setupTestFilters, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
// tslint:disable-next-line:no-require-imports

Error.stackTraceLimit = Infinity;

process.on('unhandledRejection', e => {
  throw e;
});

setTestEnvs([{name: 'cpu', backendName: 'cpu', isDataSync: true}]);

const coreTests = 'node_modules/@tensorflow/tfjs-core/dist/tests.js';
const cpuTests = 'src/**/*_test.ts';

const runner = new jasmineCtor();
runner.loadConfig({spec_files: [cpuTests, coreTests], random: false});

const TEST_FILTERS: TestFilter[] = [];
const customInclude = (testName: string) => {
  // Exclude webworker test
  if (testName.includes('computation in worker')) {
    return false;
  }

  // Include all other tests.
  return true;
};
setupTestFilters(TEST_FILTERS, customInclude);

runner.execute();
