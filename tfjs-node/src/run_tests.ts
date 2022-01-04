/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';

import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line:no-imports-from-dist
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';
import {argv} from 'yargs';

import {NodeJSKernelBackend} from './nodejs_kernel_backend';

Error.stackTraceLimit = Infinity;

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
// tslint:disable-next-line:no-require-imports

process.on('unhandledRejection', e => {
  throw e;
});

jasmine_util.setTestEnvs([{
  name: 'test-tensorflow',
  backendName: 'tensorflow',
  flags: {},
  isDataSync: true
}]);

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
  // tslint:disable-next-line:max-line-length
  'maxPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 default dimRoundingMode',
  'maxPool test-tensorflow {} x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor',
  'maxPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=floor',
  'maxPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=round',
  'maxPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=ceil',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'avgPool test-tensorflow {} x=[3,3,1] f=[3,3] s=1 p=explicit',
  // tslint:disable-next-line:max-line-length
  'avgPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 default dimRoundingMode',
  'avgPool test-tensorflow {} x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor',
  'avgPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=floor',
  'avgPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=round',
  'avgPool test-tensorflow {} x=[2,2,3] f=[2,2] s=3 p=1 dimRoundingMode=ceil',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'avgPool test-tensorflow {} gradient x=[3,3,1] f=[3,3] s=1 p=explicit',
  // tslint:disable-next-line:max-line-length
  'avgPool3d test-tensorflow {} x=[1,2,2,2,1] f=[2,2,2] s=1 p=1 roundingMode=floor',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'maxPool test-tensorflow {} x=[3,3,1] f=[3,3] s=1 p=explicit',
  'maxPoolBackprop test-tensorflow {} gradient x=[3,3,1] f=3 s=1 p=explicit',
  // tslint:disable-next-line:max-line-length
  'maxPool3d test-tensorflow {} x=[1,2,2,2,1] f=[2,2,2] s=1 p=1 roundingMode=floor',
  // libtensorflow doesn't support 6D ArgMax yet.
  'argmax test-tensorflow {} 6D, axis=0',
  'diag test-tensorflow {} complex',
  'diag test-tensorflow {} bool',
  // See https://github.com/tensorflow/tfjs/issues/1891
  'conv2d test-tensorflow {} x=[2,1,2,2] f=[1,1,1,1] s=1 d=1 p=0 NCHW',
  'conv2d test-tensorflow {} x=[1,2,2] f=[2,2,1,1] s=1 d=1 p=explicit NCHW',
  'conv2d test-tensorflow {} x=[1,2,2] f=[2,2,1,1] s=1 d=1 p=same NCHW',
  'conv2d test-tensorflow {} x=[2,2,2] f=[2,2,2,1] s=1 d=1 p=same NCHW',
  'conv2d test-tensorflow {} x=[2,1,2,2] f=[2,2,1,1] s=1 d=1 p=same NCHW',
  'conv2d test-tensorflow {} gradient x=[1,1,3,3] f=[2,2,1,1] s=1 p=0 NCHW',
  'conv2d test-tensorflow {} gradient x=[2,1,3,3] f=[2,2,1,1] s=1 p=0 NCHW',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'conv2dTranspose test-tensorflow {} input=3x3x1,d2=1,f=2,s=2,p=explicit',
  // tslint:disable-next-line:max-line-length
  'conv2dTranspose test-tensorflow {} gradient input=[1,3,3,1] f=[2,2,2,1] s=[1,1] p=explicit',
  'maxPoolWithArgmax',
  'rotate',
  // FIXME: unique NaN-handling is inconsistent between TFJS and TFPY
  'unique test-tensorflow {} 1d tensor with NaN and Infinity',
  // libtensorflow does not yet support tf.matmul with broadcast
  'broadcast with unequal batch dims',
  'broadcast batch shape',
  'broadcast with unequal ranks',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'pool test-tensorflow {} max x=[3,3,1] f=[3,3] s=1 d=1 p=explicit',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} max x=[3,3,1] f=[3,3] s=3 d=1 p=explicit defualt dimRoundingMode',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} max x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=floor',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} max x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=round',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} max x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=ceil',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} avg x=[3,3,1] f=[3,3] s=3 d=1 p=explicit defualt dimRoundingMode',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} avg x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=floor',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} avg x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=round',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} avg x=[3,3,1] f=[3,3] s=3 d=1 p=explicit dimRoundingMode=ceil',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} max x=[2,2,3] f=[1,1] s=2 p=1 fractional outputs default rounding',
  // Node backend which uses TF 2.4.0 doesn't support explicit padding
  'pool test-tensorflow {} avg x=[3,3,1] f=[3,3] s=1 d=1 p=explicit',
  // tslint:disable-next-line:max-line-length
  'pool test-tensorflow {} avg x=[2,2,3] f=[1,1] s=2 p=1 fractional outputs default rounding',
  // not available in tf yet.
  'denseBincount',
  // only available in tf addon.
  'image.transform',
  'sign test-tensorflow {} basic',
  'sign test-tensorflow {} does not propagate NaNs',
  'sign test-tensorflow {} accepts a tensor-like object',
  // Node kernel for einsum is yet to be implemented.
  // See: ttps://github.com/tensorflow/tfjs/issues/2349
  'einsum',
  'sparseFillEmptyRows',
  'sparseReshape',
  'sparseSegmentMean',
  'sparseSegmentSum',
  'stringNGrams',
  'stringSplit',
  'stringToHashBucketFast',
  'greaterEqual test-tensorflow {} should support string comparison',
  'greater test-tensorflow {} should support string comparison',
  'lessEqual test-tensorflow {} should support string comparison',
  'less test-tensorflow {} should support string comparison',
  // int32 input for the following ops not supported by TF 2.4.0.
  'leakyrelu test-tensorflow {} int32',
  'step kernel test-tensorflow {} with int32 tensor',
];

if (process.platform === 'win32') {
  // Ignore tensorboard on windows because tmp folder cleanup used in tests
  // fails with tmp folder cleanup:
  // https://github.com/tensorflow/tfjs/issues/1692
  IGNORE_LIST.push('tensorboard');
  IGNORE_LIST.push('tensorBoard');
  // Windows has two failing tests:
  // https://github.com/tensorflow/tfjs/issues/598
  IGNORE_LIST.push('clip test-tensorflow {} propagates NaNs');
  IGNORE_LIST.push(
      'maxPool test-tensorflow {} [x=[3,3,1] f=[2,2] s=1 ignores NaNs');
}

const runner = new jasmineCtor();
runner.loadConfig({spec_files: ['src/**/*_test.ts'], random: false});
// Also import tests from core.
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/tests';

if (process.env.JASMINE_SEED) {
  runner.seed(process.env.JASMINE_SEED);
}

const env = jasmine.getEnv();

const grepRegex = new RegExp(argv.grep as string);

// Filter method that returns boolean, if a given test should return.
env.specFilter = spec => {
  // Filter based on the grep flag.
  if (!grepRegex.test(spec.getFullName())) {
    return false;
  }
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
console.log(`Running tests against TensorFlow: ${
    (tf.backend() as NodeJSKernelBackend).binding.TF_Version}`);
runner.execute();
