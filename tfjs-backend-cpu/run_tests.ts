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

// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as fs from 'fs';

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
// tslint:disable-next-line:no-require-imports

Error.stackTraceLimit = Infinity;

process.on('unhandledRejection', e => {
  throw e;
});

setTestEnvs([{name: 'cpu', backendName: 'cpu', isDataSync: true}]);

const IGNORE_LIST: string[] = [
  // Exclude webworker tests.
  'computation in worker'
];

const coreTests = 'node_modules/@tensorflow/tfjs-core/dist/**/*_test.js';
const cpuTests = 'src/**/*_test.ts';

const runner = new jasmineCtor();
runner.loadConfig({spec_files: [coreTests, cpuTests], random: false});

const env = jasmine.getEnv();

env.specFilter = spec => {
  for (let i = 0; i < IGNORE_LIST.length; ++i) {
    if (spec.getFullName().indexOf(IGNORE_LIST[i]) > -1) {
      fs.appendFileSync(
          'testlog.txt', `${spec.getFullName()}_____skipped.\r\n`);
      return false;
    }
  }

  fs.appendFileSync('testlog.txt', `${spec.getFullName()}.\r\n`);
  return true;
};

runner.execute();
