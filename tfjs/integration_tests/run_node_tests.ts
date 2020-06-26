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

// tslint:disable-next-line: no-imports-from-dist
import * as jasmineUtil from '@tensorflow/tfjs-core/dist/jasmine_util';
import * as fs from 'fs';

// tslint:disable-next-line:no-any
export function runTests(jasmineUtil: any, filename: string): void {
  // tslint:disable-next-line:no-require-imports
  const jasmineCtor = require('jasmine');

  Error.stackTraceLimit = Infinity;

  process.on('unhandledRejection', e => {
    throw e;
  });

  jasmineUtil.setTestEnvs(
      [{name: 'node', factory: jasmineUtil.CPU_FACTORY, features: {}}]);

  const runner = new jasmineCtor();
  runner.loadConfig({
    frameworks: ['jasmine', 'karma-typescript'],
    spec_files: [filename],
    random: false
  });
  runner.execute();
}

const filenameIndex = process.argv.indexOf('--filename');
if (filenameIndex === -1 || filenameIndex + 1 >= process.argv.length) {
  throw new Error('Missing value for flag --filename');
}

const filename = process.argv[filenameIndex + 1];

if (!fs.existsSync(filename)) {
  throw new Error(`File does not exist: ${filename}`);
}

runTests(jasmineUtil, filename);
