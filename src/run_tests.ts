/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';

Error.stackTraceLimit = Infinity;

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
// tslint:disable-next-line:no-require-imports
import {nodeBackend} from './ops/op_utils';

process.on('unhandledRejection', e => {
  throw e;
});

jasmine_util.setTestEnvs(
    [{name: 'test-tensorflow', backendName: 'tensorflow', flags: {}}]);

const IGNORE_LIST: string[] = [
  // Always ignore version tests:
  'version version',
  'unreliable is true due to both auto gc and string tensors',
  'unreliable is true due to auto gc',
  // See https://github.com/tensorflow/tfjs/issues/161
  'depthwiseConv2D',  // Requires space_to_batch() for dilation > 1.
  'separableConv2d',  // Requires space_to_batch() for dilation > 1.
  'complex64 memory',
  // See https://github.com/tensorflow/tfjs-core/pull/1270
  'depthToSpace test-tensorflow {} throws when blocksize < 2',
  // tslint:disable-next-line:max-line-length
  'depthToSpace test-tensorflow {} throws when CPU backend used with data format NCHW',
  // See https://github.com/tensorflow/tfjs/issues/806
  'scatterND test-tensorflow {} should work for 2d',
  'scatterND test-tensorflow {} should work for simple 1d',
  'scatterND test-tensorflow {} should work for multiple 1d',
  'scatterND test-tensorflow {} should sum the duplicated indices',
  'scatterND test-tensorflow {} should work for tensorLike input',
  // https://github.com/tensorflow/tfjs/issues/1077
  'maxPool test-tensorflow {} x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor',
  'avgPool test-tensorflow {} x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor',
  // libtensorflow doesn't support 6D ArgMax yet.
  'Reduction: argmax test-tensorflow {} 6D, axis=0'
];

// Windows has two failing tests:
// https://github.com/tensorflow/tfjs/issues/598
if (process.platform === 'win32') {
  IGNORE_LIST.push('clip test-tensorflow {} propagates NaNs');
  IGNORE_LIST.push(
      'maxPool test-tensorflow {} [x=[3,3,1] f=[2,2] s=1 ignores NaNs');
}

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

// TODO(kreeger): Consider moving to C-code.
console.log(
    `Running tests against TensorFlow: ${nodeBackend().binding.TF_Version}`);
runner.execute();
