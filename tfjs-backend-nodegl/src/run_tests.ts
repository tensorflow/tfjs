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

// We import index.ts so that the Node backend gets registered.
import './index';

import * as tf from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';

Error.stackTraceLimit = Infinity;

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');

process.on('unhandledRejection', e => {
  throw e;
});

jasmine_util.setTestEnvs(
    [{name: 'test-tensorflow', backendName: 'headless-nodegl', flags: {}}]);

const IGNORE_LIST: string[] = [
  // https://github.com/tensorflow/tfjs/issues/1711
  'time cpu test-tensorflow {} simple upload',
  // TODO(kreeger): File issue: bad uniform in input.uniformValues.
  'sparseToDense test-tensorflow {} should work with 0-sized tensors',
  // TODO(kreeger): File issue: fromPixels doesn't have data field.
  // tslint:disable:max-line-length
  'fromPixels, mock canvas test-tensorflow {} accepts a canvas-like element, numChannels=4',
  'fromPixels, mock canvas test-tensorflow {} accepts a canvas-like element'
];

const runner = new jasmineCtor();
runner.loadConfig({
  spec_files: [
    'src/**/*_test.ts', 'node_modules/@tensorflow/tfjs-core/dist/**/*_test.js'
  ],
  random: false
});

if (process.env.JASMINE_SEED) {
  runner.seed(process.env.JASMINE_SEED);
}

const env = jasmine.getEnv();

// Filter method that returns boolean, if a given test should return.
env.specFilter = spec => {
  // Return false (skip the test) if the test is in the ignore list.
  for (let i = 0; i < IGNORE_LIST.length; ++i) {
    if (spec.getFullName().indexOf(IGNORE_LIST[i]) > -1) {
      return false;
    }
  }
  // Otherwise run the test.
  return true;
};

console.log(`Running tests with the following GL info`);
const gl = (tf.backend() as tf.webgl.MathBackendWebGL).getGPGPUContext().gl;
console.log(`  GL_VERSION: ${gl.getParameter(gl.VERSION)}`);
console.log(`  GL_RENDERER: ${gl.getParameter(gl.RENDERER)}`);

runner.execute();
