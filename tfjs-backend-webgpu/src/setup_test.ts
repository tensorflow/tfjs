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
  {
    startsWith: 'abs ',
    excludes: [
      'complex64',  // Kernel 'ComplexAbs' not registered.
      '5D',         // Rank 5 is not yet implemented.
      '6D',         // Rank 5 is not yet implemented.
      'gradient',   // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'add ',
    excludes: [
      '6D',        // Rank 6 is not yet implemented.
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'addN',
  },
  {
    startsWith: 'argmax',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'argmin',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'avgPool',
    excludes: [
      'gradient',   // Not yet implemented.
      'avgPool3d',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'batchNorm',
  },
  {
    startsWith: 'batchToSpaceND',
    excludes: [
      'tensor3d', 'tensor4d', 'gradient',
      'accepts a tensor-like object',  // tensor6d not yet implemented
    ]
  },
  {
    startsWith: 'ceil',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'clipByValue',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'complex64',
  },
  {
    startsWith: 'concat',
    excludes: [
      'concat a large number of tensors',  // The number of storage buffers
                                           // exceeds the maximum per-stage
                                           // limit.
      'gradient',                          // split not yet implemented.
    ]
  },
  {
    startsWith: 'conv2d',
    excludes: [
      'NCHW',      // Not yet implemented.
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'cos',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'cropAndResize',
  },
  {
    startsWith: 'depthToSpace',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'depthwise',
    excludes: [
      'gradient',  // depthwiseConv2DDerInput not yet implemented.
    ]
  },
  {
    startsWith: 'div',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'dot',
  },
  {
    startsWith: 'einsum',
    excludes: [
      '4d tensors',               // rank 5 is not yet supported.
      '4d tensor and 3d tensor',  // rank 5 is not yet supported.
    ]
  },
  {
    startsWith: 'elu',
    excludes: [
      'selu',        // Not yet implemented.
      'derivative',  // gradient function not found.
      'gradient'     // gradient function not found.
    ]
  },
  {
    startsWith: 'equal ',
  },
  {
    startsWith: 'exp ',
    excludes: [
      'int32',  // precision problem.
    ]
  },
  {
    startsWith: 'expandDims',
  },
  {
    startsWith: 'expm1',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'fill ',
    excludes: [
      '5D',  // Rank 5 is not yet supported.
    ]
  },
  {
    startsWith: 'flipLeftRight',
  },
  {
    startsWith: 'floor ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'floorDiv',
  },
  {
    startsWith: 'fromPixels',
  },
  {
    startsWith: 'fromPixelsAsync',
  },
  {
    startsWith: 'fused conv2d',
    excludes: [
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // conv2dDerInput not yet
                                                   // implemented
      'backProp',   // Conv2DBackpropFilter not yet
                    // implemented
      'leakyrelu',  // Not yet implemented
    ]
  },
  {
    startsWith: 'fused depthwiseConv2D',
    excludes: [
      'gradient',   // gradient function not found.
      'leakyrelu',  // Not yet implemented
    ]
  },
  {
    startsWith: 'fused matmul',
    excludes: [
      'leakyrelu',  // Not yet implemented.
      'gradient',   // Not yet implemented.
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
    startsWith: 'gatherND ',
  },
  {
    startsWith: 'greater',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'image.transform',
  },
  {
    startsWith: 'less',
  },
  {
    startsWith: 'log ',
    excludes: [
      '6D',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'logicalAnd ',
  },
  {
    startsWith: 'logicalNot',
    excludes: [
      'Tensor6D',  // Not yet implemented.
      'gradient'   // gradient function not found.
    ]
  },
  {
    startsWith: 'leakyrelu',
    excludes: [
      'gradients: Tensor2D',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'matmul',
    excludes: [
      'gradient',                        // Various: sum not yet implemented.
      'has zero in its shape',           // Test times out.
      'valueAndGradients',               // backend.sum() not yet implemented.
      'upcasts when dtypes dont match',  // GLSL compilation failed
      'broadcast',  // matmul broadcasting not yet implemented.
    ]
  },
  {
    startsWith: 'max ',
    excludes: [
      '6D', 'gradient',
      'AdamaxOptimizer',    // gradient function not found.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    startsWith: 'maxPool',
    excludes: [
      'maxPoolBackprop',   // Not yet implemented.
      'maxPool3d',         // Not yet implemented.
      'maxPoolWithArgmax'  // Not yet implemented.
    ]
  },
  {
    startsWith: 'maximum',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'mean',
    excludes: [
      'gradient',
      'meanSquaredError',
    ]
  },
  {
    startsWith: 'memory',
  },
  {
    startsWith: 'min ',
    excludes: [
      'gradient',
      'stft',  // FFT' not registered.
    ]
  },
  {
    startsWith: 'minimum',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'mirrorPad',
    excludes: [
      'grad',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'mul ',
    excludes: [
      'broadcast',  // Various: Actual != Expected, compile fails, etc.
      'gradient',   // gradient function not found.
    ]
  },
  {
    startsWith: 'neg',
  },
  {
    startsWith: 'nonMaxSuppression',
    excludes: [
      'NonMaxSuppressionPadded'  // NonMaxSuppressionV4 not yet implemented.
    ]
  },
  {
    startsWith: 'notEqual ',
  },
  {
    startsWith: 'onesLike',
    excludes: [
      '5D',       // rank 5 is not yet supported.
      '6D',       // rank 6 is not yet supported.
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'pad ',
    excludes: [
      'grad'  // gradient function not found.
    ]
  },
  {
    startsWith: 'pool',
    excludes: [
      'avg x=[',                          // Unsupported 6D shape.
      'max x=[4,3,1] f=[2,2] s=1 d=2',    // Unsupported 6D shape.
      'max x=[2,4,4,1] f=[2,2] s=1 d=2',  // Unsupported 6D shape.
      'poolBackprop',  // maxPoolBackprop not yet implemented.
    ]
  },
  {
    startsWith: 'pow ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'prod',
  },
  {
    startsWith: 'range',
    excludes: [
      'bincount',           // Not yet implemented.
      'denseBincount',      // Not yet implemented.
      'oneHot',             // Not yet implemented.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    startsWith: 'Reduction: max',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'Reduction: min',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'Reduction: sum',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'relu',
    excludes: [
      'valueAndGradients',  // gradient function not found.
      '5D',                 // Rank 5 is not yet implemented.
      '6D',                 // Rank 5 is not yet implemented.
      'propagates NaNs',    // Arrays differ.
      'derivative',         // gradient function not found.
      'gradient'            // gradient function not found.
    ]
  },
  {
    startsWith: 'resizeBilinear',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'resizeNearest',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'rotateWithOffset',
  },
  {
    startsWith: 'rsqrt',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'scatterND',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'sigmoid ',
    excludes: [
      '6D'  // Not yet implemented.
    ]
  },
  {
    startsWith: 'sin',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'slice ',
    excludes: [
      '5D',                  // Rank 5 is not yet implemented.
      'slice5d',             // Rank 5 is not yet implemented.
      '6D',                  // Rank 6 is not yet implemented.
      'slice6d',             // Rank 6 is not yet implemented.
      'strided slice with',  // Rank 6 is not yet implemented.
    ]
  },
  {
    startsWith: 'softmax',
    excludes: [
      'gradient',
      'MEAN',
      'Weighted - Reduction.SUM_BY_NONZERO_WEIGHTS',
    ]
  },
  {
    startsWith: 'spaceToBatchND',
    excludes: [
      'tensor4d',
      'gradient',
      'accepts a tensor-like object',
    ]
  },
  {
    startsWith: 'sparseToDense',
    excludes: [
      // TODO: Fix 0-sized buffer binding on WebGPU
      '0-sized',  // Not yet implemented.
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'split',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'sqrt',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'square',
    excludes: [
      '5D',          // Rank 5 is not yet implemented.
      '6D',          // Rank 6 is not yet implemented.
      'dilation2d',  // 'dilation2d' not yet implemented.
      'gradient',
    ]
  },
  {
    startsWith: 'stack',
    excludes: [
      'grad of unstack axis=0',  // Remove this when grad is fixed in unstack.
      'gradient with clones',    // Remove this when grad is fixed in unstack.
      'grad of unstack axis=1',  // Remove this when grad is fixed in unstack.
    ]
  },
  {
    startsWith: 'stridedSlice',
    excludes: [
      'strided slice with several new axes',  // Rank 6 is not yet implemented.
      'strided slice with new axes and',      // Rank 6 is not yet implemented.
    ]
  },
  {
    startsWith: 'stringNGrams ',
  },
  {
    startsWith: 'sub ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'subtract ',
  },
  {
    startsWith: 'sum ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'tanh',
    excludes: [
      'grad',  // gradient function not found.
    ]
  },
  {
    startsWith: 'tensor',
    excludes: [
      'grad',        // gradient function not found.
      'bool tensor'  // Expected object not to have properties.
    ]
  },
  {
    startsWith: 'tile ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'topk',
  },
  {
    startsWith: 'transpose',
    excludes: [
      'oneHot',  // Not yet implemented.
      'fused',   // Not yet implemented.
      '5D',      // Rank 5 is not yet implemented.
      '6D',      // Rank 5 is not yet implemented.
      'gradient',
    ]
  },
  {
    startsWith: 'unstack',
    excludes: [
      'grad of unstack axis=0',
      'gradient with clones',
      'grad of unstack axis=1',
    ]
  },
  {
    startsWith: 'where ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'zerosLike',
    excludes: [
      '5D',       // rank 5 is not yet supported.
      '6D',       // rank 6 is not yet supported.
      'gradient'  // gradient function not found.
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
