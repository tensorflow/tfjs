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
    include: 'less',
  },
  {
    include: 'clip',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    include: 'greater',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    include: 'div',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    include: 'depthwise',
    excludes: [
      'gradient',   // depthwiseConv2DDerInput not yet implemented.
      'leakyrelu',  // Not yet implemented.
    ]
  },
  {
    include: 'fused conv2d',
    excludes: [
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // conv2dDerInput not yet
                                                   // implemented
      'backProp',   // Conv2DBackpropFilter not yet
                    // implemented
      'leakyrelu',  // Not yet implemented
    ]
  },
  {
    include: 'fromPixels',
  },
  {
    include: 'fromPixelsAsync',
  },
  {
    include: 'nonMaxSuppression',
    excludes: [
      'NonMaxSuppressionPadded'  // NonMaxSuppressionV4 not yet implemented.
    ]
  },
  {
    include: 'argmax',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    include: 'concat',
    excludes: [
      'concat a large number of tensors',  // The number of storage buffers
                                           // exceeds the maximum per-stage
                                           // limit.
      'gradient',                          // split not yet implemented.
    ]
  },
  {
    include: 'transpose',
    excludes: [
      'oneHot',  // Not yet implemented.
      'fused',   // Not yet implemented.
      '5D',      // Rank 5 is not yet implemented.
      '6D',      // Rank 5 is not yet implemented.
      'gradient',
    ]
  },
  {
    include: 'relu',
    excludes: [
      'valueAndGradients',  // gradient function not found.
      '5D',                 // Rank 5 is not yet implemented.
      '6D',                 // Rank 5 is not yet implemented.
      'propagates NaNs',    // Arrays differ.
      'derivative',         // gradient function not found.
      'gradient',           // gradient function not found.
      'leakyrelu'           // Not yet implemented.
    ]
  },
  {
    startsWith: 'leakyrelu',
  },
  {
    include: 'elu',
    excludes: [
      'selu',        // Not yet implemented.
      'derivative',  // gradient function not found.
      'gradient'     // gradient function not found.
    ]
  },
  {
    include: 'resizeBilinear',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    include: 'ceil',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {
    include: 'floor ',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {include: 'floor divide ', excludes: []},
  {
    include: 'rsqrt',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {
    include: 'expm1',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {
    include: 'fused',
    excludes: [
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // conv2dDerInput not yet
                                                   // implemented.
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias',  // conv2dDerInput
                                                             // not yet
                                                             // implemented.
      'leakyrelu',  // Not yet implemented.
    ]
  },
  {
    include: 'maxPool',
    excludes: [
      'maxPoolBackprop',   // Not yet implemented.
      'maxPool3d',         // Not yet implemented.
      'maxPoolWithArgmax'  // Not yet implemented.
    ]
  },
  {
    include: 'avgPool',
    excludes: [
      'x=[2,2,1] f=[2,2] s=1 p=same',  // Pool3D not yet implemented.
      'gradient',                      // Not yet implemented.
      'avgPoolBackprop',               // Not yet implemented.
      'avgPool3d',                     // Not yet implemented.
      'avgPoolWithArgmax'              // Not yet implemented.
    ]
  },
  {
    include: 'pool',
    excludes: [
      'avg x=[',                          // Unsupported 6D shape.
      'max x=[4,3,1] f=[2,2] s=1 d=2',    // Unsupported 6D shape.
      'max x=[2,4,4,1] f=[2,2] s=1 d=2',  // Unsupported 6D shape.
      'poolBackprop',  // maxPoolBackprop not yet implemented.
    ]
  },
  {
    include: 'matmul',
    excludes: [
      'gradient',                        // Various: sum not yet implemented.
      'has zero in its shape',           // Test times out.
      'valueAndGradients',               // backend.sum() not yet implemented.
      'upcasts when dtypes dont match',  // GLSL compilation failed
      'broadcast',  // matmul broadcasting not yet implemented.
      'leakyrelu',  // Not yet implemented.
    ]
  },
  {
    include: 'dot',
  },
  {
    include: 'expandDims',
  },
  {
    include: 'memory test',
  },
  {
    include: 'add ',
    excludes: [
      '6D',        // Rank 6 is not yet implemented.
      'gradient',  // gradient function not found.
    ]
  },
  {include: 'addN', excludes: []},
  {startsWith: 'floorDiv ', excludes: []},
  {
    startsWith: 'sub ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {startsWith: 'subtract ', excludes: []},
  {
    include: 'square',
    excludes: [
      '5D',          // Rank 5 is not yet implemented.
      '6D',          // Rank 6 is not yet implemented.
      'dilation2d',  // 'dilation2d' not yet implemented.
      'gradient',
    ]
  },
  {
    include: 'slice ',
    excludes: [
      '5D',                  // Rank 5 is not yet implemented.
      'slice5d',             // Rank 5 is not yet implemented.
      '6D',                  // Rank 6 is not yet implemented.
      'slice6d',             // Rank 6 is not yet implemented.
      'strided slice with',  // Rank 6 is not yet implemented.
    ]
  },
  {
    include: 'stridedSlice',
    excludes: [
      'strided slice with several new axes',  // Rank 6 is not yet implemented.
      'strided slice with new axes and',      // Rank 6 is not yet implemented.
    ]
  },
  {
    include: 'mul ',
    excludes: [
      'broadcast',  // Various: Actual != Expected, compile fails, etc.
      'gradient',   // gradient function not found.
    ]
  },
  {
    include: 'conv2d',
    excludes: [
      'NCHW',       // Not yet implemented.
      'gradient',   // gradient function not found.
      'leakyrelu',  // Not yet implemented.
    ]
  },
  {
    include: 'mirrorPad',
    excludes: [
      'gradient',  // Not yet implemented.
      'grad',      // Not yet implemented.
    ]
  },
  {
    startsWith: 'pad ',
    excludes: [
      'grad'  // gradient function not found.
    ]
  },
  {
    startsWith: 'fill ',
    excludes: [
      '5D',  // Rank 5 is not yet supported.
    ]
  },
  {
    include: 'Reduction: max',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    include: 'Reduction: min',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    include: 'Reduction: sum',
    excludes: [
      '5D',        // Rank 5 is not yet implemented.
      '6D',        // Rank 5 is not yet implemented.
      'gradient',  // zerosLike not yet implemented.
    ]
  },
  {
    startsWith: 'abs ',
    excludes: [
      'complex64',           // Kernel 'ComplexAbs' not registered.
      '5D',                  // Rank 5 is not yet implemented.
      '6D',                  // Rank 5 is not yet implemented.
      'gradient',            // zerosLike not yet implemented.
      'absoluteDifference',  // absoluteDifference not yet implemented
    ]
  },
  {
    include: 'cropAndResize',
    excludes: [
      '2x2to3x3-NoCrop',  // The operation failed for an operation-specific
                          // reason
      'MultipleBoxes-DifferentBoxes',  // TimeOut
    ]
  },
  {
    include: 'batchNorm',
    excludes: [
      'gradient',
    ]
  },
  {
    include: 'batchToSpaceND',
    excludes: [
      'tensor3d', 'tensor4d', 'gradient',
      'accepts a tensor-like object',  // tensor6d not yet implemented
    ]
  },
  {
    include: 'spaceToBatchND',
    excludes: [
      'tensor4d',
      'gradient',
      'accepts a tensor-like object',
    ]
  },
  {
    include: 'softmax',
    excludes: [
      'gradient',
      'MEAN',
      'Weighted - Reduction.SUM_BY_NONZERO_WEIGHTS',
    ]
  },
  {
    include: 'minimum',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {
    include: 'maximum',
    excludes: [
      'gradients: Scalar',
      'gradient with clones',
      'gradients: Tensor1D',
      'gradients: Tensor2D',
    ]
  },
  {
    include: 'stack',
    excludes: [
      'grad of unstack axis=0',  // Remove this when grad is fixed in unstack.
      'gradient with clones',    // Remove this when grad is fixed in unstack.
      'grad of unstack axis=1',  // Remove this when grad is fixed in unstack.
    ]
  },
  {
    include: 'unstack',
    excludes: [
      'grad of unstack axis=0',
      'gradient with clones',
      'grad of unstack axis=1',
    ]
  },
  {
    include: 'complex64',
  },
  {
    include: 'zerosLike',
    excludes: [
      '5D',       // rank 5 is not yet supported.
      '6D',       // rank 6 is not yet supported.
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'onesLike',
    excludes: [
      '5D',       // rank 5 is not yet supported.
      '6D',       // rank 6 is not yet supported.
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'gather',
    excludes: [
      'throws when index is out of bound',
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'max',
    excludes: [
      '6D', 'gradient',
      'AdamaxOptimizer',    // gradient function not found.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    include: 'mean',
    excludes: [
      'gradient',
      'meanSquaredError',
    ]
  },
  {
    include: 'min',
    excludes: [
      'gradient',
      'stft',  // FFT' not registered.
    ]
  },
  {
    include: 'prod',
  },
  {
    include: 'einsum',
    excludes: [
      '4d tensors',               // rank 5 is not yet supported.
      '4d tensor and 3d tensor',  // rank 5 is not yet supported.
    ]
  },
  {
    include: 'sum',
    excludes: [
      'gradient',
      'cumsum',  // 'Cumsum' not registered.
    ]
  },
  {
    include: 'range',
    excludes: [
      'bincount',           // Not yet implemented.
      'denseBincount',      // Not yet implemented.
      'oneHot',             // Not yet implemented.
      'sparseSegmentMean',  // 'SparseSegmentMean' not registered.
    ]
  },
  {
    include: 'resizeNearest',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'split',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'sqrt',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'depthToSpace',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'logicalNot',
    excludes: [
      'Tensor6D',  // Not yet implemented.
      'gradient'   // gradient function not found.
    ]
  },
  {
    include: 'flipLeftRight',
  },
  {
    include: 'rotateWithOffset',
  },
  {
    include: 'topk',
  },
  {
    include: 'sparseToDense',
    excludes: [
      // TODO: Fix 0-sized buffer binding on WebGPU
      '0-sized',  // Not yet implemented.
      'gradient'  // gradient function not found.
    ]
  },
  {
    include: 'scatterND',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'logicalAnd ',
  },
  {
    startsWith: 'stringNGrams ',
  },
  {
    startsWith: 'tile ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'pow ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'equal ',
  },
  {
    startsWith: 'notEqual ',
  },
  {startsWith: 'gatherND '},
  {include: 'image.transform'},
  {
    startsWith: 'where ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {startsWith: 'memory'},
  {
    startsWith: 'sin',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'cos',
    excludes: [
      'gradient'  // gradient function not found.
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
    startsWith: 'tanh',
    excludes: [
      'grad',  // gradient function not found.
    ]
  }
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
