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

import * as tfc from '@tensorflow/tfjs-core';
import {bindTensorFlowBackend} from './index';

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
bindTensorFlowBackend();

tfc.test_util.setBeforeAll(() => {});
tfc.test_util.setAfterAll(() => {});
tfc.test_util.setBeforeEach(() => {});
tfc.test_util.setAfterEach(() => {});
tfc.test_util.setTestEnvFeatures([{BACKEND: 'tensorflow'}]);

const IGNORE_LIST: string[] = [
  // Backend methods.
  'memory',
  'variable',  // Depends on backend.memory
  'debug',     // Depends on backend.time
  'tidy',      // Depeonds on backend.memory

  // Optimizers.
  'RMSPropOptimizer',
  'MomentumOptimizer',
  'AdagradOptimizer',
  'AdamaxOptimizer',
  'AdamOptimizer',
  'SGDOptimizer',
  'AdadeltaOptimizer',
  'optimizer',

  // Unimplemented ops.
  'clip',
  'leakyRelu',
  'elu',
  'expm1',
  'log1p',
  'resizeBilinear',
  'argmin',
  'argmax',
  'avgPool',
  'multinomial',
  'localResponseNormalization',
  'logicalXor',
  'depthwiseConv2D',
  'conv1d',
  'conv2dTranspose',
  'conv2d',
  'atan2',
  'squaredDifference',
  'prelu',
  'batchNormalization2D',
  'batchNormalization3D',
  'batchNormalization4D',
  'tile',
  'rsqrt',
  'sign',
  'acosh',
  'asinh',
  'atanh',
  'reciprocal',
  'round',
  'separableConv2d',
  'mod',
  'maxPool',
  'minPool',

  // Ops with bugs. Some are higher-level ops.
  'norm',  // Depends on tf.pow being fixed.
  'oneHot',
  'gather',
  'pow',
  'absoluteDifference',
  'computeWeightedLoss',

  // Depends on ops being fixed first.
  'gradients',
  'customGradient',
];

const runner = new jasmineCtor();
runner.loadConfig({
  spec_files: [
    'src/**/*_test.ts', 'node_modules/@tensorflow/tfjs-core/dist/**/*_test.js'
  ]
});

const env = jasmine.getEnv();

// Filter method that returns boolean, if a given test should return.
env.specFilter = spec => {
  // Return false (skip the test) if the test is in the ignore list.
  for (let i = 0; i < IGNORE_LIST.length; ++i) {
    if (spec.getFullName().startsWith(IGNORE_LIST[i])) {
      return false;
    }
  }
  // Otherwise run the test.
  return true;
};

runner.execute();
