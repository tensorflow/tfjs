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

// Register the backend.
import './index';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import './backend_webgpu_test_registry';
// tslint:disable-next-line: no-imports-from-dist
import {parseTestEnvFromKarmaFlags, setTestEnvs, setupTestFilters, TEST_ENVS, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

const TEST_FILTERS: TestFilter[] = [
  // skip test cases include gradients webgpu
  {
    include: 'gradients webgpu',
    excludes: ['webgpu '],
  },

  // skip specific test cases for supported kernels
  {
    startsWith: 'abs ',
    excludes: [
      'complex64',  // Kernel 'ComplexAbs' not registered.
      'gradient',   // Step kernel not yet implemented.
    ]
  },
  {
    startsWith: 'atan2 ',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'avgPool ',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'batchToSpaceND ',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'conv2d ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'conv2dTranspose ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'cumprod ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'prod ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'cumsum ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'elu ',
    excludes: [
      'selu',        // Not yet implemented.
      'derivative',  // gradient function not found.
      'gradient'     // gradient function not found.
    ]
  },
  {
    startsWith: 'exp ',
    excludes: [
      'int32',  // TODO: fix precision problem.
    ]
  },
  {
    startsWith: 'fused conv2d ',
    excludes: [
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // conv2dDerInput not yet
                                                   // implemented
      'backProp',  // Conv2DBackpropFilter not yet
                   // implemented
    ]
  },
  {
    startsWith: 'fused depthwiseConv2D ',
    excludes: [
      'gradient',  // DepthwiseConv2dNativeBackpropInput
    ]
  },
  {
    startsWith: 'fused matmul ',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'gather ',
    excludes: [
      'throws when index is out of bound',
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'matmul',
    excludes: [
      'has zero in its shape',  // Test times out.
      'valueAndGradients',      // backend.sum() not yet implemented.
    ]
  },
  {
    startsWith: 'maxPool ',
    excludes: [
      'maxPoolBackprop',   // Not yet implemented.
      'maxPool3d',         // Not yet implemented.
      'maxPoolWithArgmax'  // Not yet implemented.
    ]
  },
  {
    startsWith: 'max ',
    excludes: [
      'AdamaxOptimizer',    // gradient function not found.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    startsWith: 'mean ',
    excludes: [
      'meanSquaredError',
    ]
  },
  {
    startsWith: 'min ',
    excludes: [
      'stft',  // FFT' not registered.
    ]
  },
  {
    startsWith: 'mul ',
    excludes: [
      'broadcast',  // Various: Actual != Expected, compile fails, etc.
    ]
  },
  {
    startsWith: 'nonMaxSuppression ',
    excludes: [
      'NonMaxSuppressionPadded'  // NonMaxSuppressionV4 not yet implemented.
    ]
  },
  {
    startsWith: 'pool ',
    excludes: [
      'poolBackprop',  // maxPoolBackprop not yet implemented.
    ]
  },
  {
    startsWith: 'prod ',
    excludes: [
      'gradients',  // Not yet implemented
    ]
  },
  {
    startsWith: 'range ',
    excludes: [
      'bincount',           // Not yet implemented.
      'denseBincount',      // Not yet implemented.
      'oneHot',             // Not yet implemented.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    startsWith: 'relu ',
    excludes: [
      'valueAndGradients',  // gradient function not found.
      'propagates NaNs',    // Arrays differ.
      'derivative',         // gradient function not found.
      'gradient'            // gradient function not found.
    ]
  },
  {
    startsWith: 'softmax ',
    excludes: [
      'MEAN',
      'Weighted - Reduction.SUM_BY_NONZERO_WEIGHTS',
    ]
  },
  {
    startsWith: 'spaceToBatchND ',
    excludes: [
      'tensor4d',
      'accepts a tensor-like object',
    ]
  },
  {
    startsWith: 'square ',
    excludes: [
      'dilation2d',  // 'dilation2d' not yet implemented.
    ]
  },
  {
    startsWith: 'squaredDifference ',
    excludes: [
      'dilation2d',  // 'dilation2d' not yet implemented.
    ]
  },
  {
    startsWith: 'tensor ',
    excludes: [
      'bool tensor'  // Expected object not to have properties.
    ]
  },
  {
    startsWith: 'transpose ',
    excludes: [
      'oneHot',  // Not yet implemented.
      'fused',   // Not yet implemented.
    ]
  },

  // exclude unsupported kernels and to be fixed cases
  {
    include: ' webgpu ',
    excludes: [
      // Not implemented kernel list.
      'avgPool3d ',
      'avgPool3dBackprop ',
      'broadcastArgs ',
      'conv2DBackpropFilter ',
      'gradient with clones, input=2x2x1,d2=1,f=1,s=1,d=1,p=same',  // Conv2DBackpropFilter
      'conv1d gradients',  // Conv2DBackpropFilter
      'conv3d ',
      'conv3dTranspose ',
      'decodeWeights ',
      'diag ',
      'dilation2d ',
      'encodeWeights ',
      'linspace ',
      'localResponseNormalization ',
      'logSigmoid ',
      'maxPool3d ',
      'maxPool3dBackprop ',
      'maxPoolBackprop ',
      'maxPoolWithArgmax ',
      'multinomial ',
      'confusionMatrix ',  // oneHot
      'poolBackprop ',
      'raggedGather ',
      'raggedRange ',
      'raggedTensorToTensor ',
      'round webgpu',
      'method otsu',  // round
      'selu ',
      'sign webgpu',
      'stft ',
      'softplus ',
      'sigmoidCrossEntropy ',
      'sparseFillEmptyRows ',
      'sparseReshape ',
      'sparseSegmentMean ',
      'sparseSegmentSum ',
      'step kernel',
      'gradients: relu6',  // Step
      'stringSplit ',
      'stringToHashBucketFast ',
      'unique ',
      'unsortedSegmentSum ',
      'valueAndGradients ',
    ]
  },
];

const customInclude = (testName: string) => {
  // Include webgpu specific tests.
  if (testName.startsWith('webgpu')) {
    return true;
  }
  return false;
};
setupTestFilters(TEST_FILTERS, customInclude);

// Allow flags to override test envs
// tslint:disable-next-line:no-any
declare let __karma__: any;
if (typeof __karma__ !== 'undefined') {
  const testEnv = parseTestEnvFromKarmaFlags(__karma__.config.args, TEST_ENVS);
  if (testEnv != null) {
    setTestEnvs([testEnv]);
  }
}

// These use 'require' because they must not be hoisted above
// the preceding snippet that parses test environments.
// Import and run tests from core.
// tslint:disable-next-line:no-imports-from-dist
// tslint:disable-next-line:no-require-imports
require('@tensorflow/tfjs-core/dist/tests');
// Import and run tests from webgl.
// tslint:disable-next-line:no-imports-from-dist
// tslint:disable-next-line:no-require-imports
require('./tests');
