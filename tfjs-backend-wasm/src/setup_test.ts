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

// Import Object.fromEntries polyfill for Safari 11
import 'core-js/es/object/from-entries';
// Import core for side effects (e.g. flag registration)
import '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/register_all_gradients';
// Register the wasm backend.
import './index';

// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs, setupTestFilters, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {setupCachedWasmPaths} from './test_util';

setTestEnvs([{name: 'test-wasm', backendName: 'wasm', isDataSync: true}]);

/**
 * Tests that have these substrings in their name will be included unless one
 * of the strings in excludes appears in the name.
 */
const TEST_FILTERS: TestFilter[] = [
  {
    startsWith: 'tensor ',
    excludes: [
      'complex',     // Complex numbers not supported yet
      'derivative',  // Gradients not yet supported.
      // Downcasting broken, see: https://github.com/tensorflow/tfjs/issues/2590
      'Tensor2D float32 -> bool', 'Tensor2D int32 -> bool'
    ]
  },
  {include: 'softmax'},
  {
    include: 'pow',
    excludes: [
      'gradient'  // zerosLike not defined yet.
    ]
  },
  {
    include: 'add ',
    excludes: [
      'gradient',                        // Gradient is missing.
      'upcasts when dtypes dont match',  // Uses the 'complex' dtype.
      'complex',                         // Complex numbers not supported yet
    ]
  },
  {include: 'depthToSpace'},
  {include: 'avgPool '},
  {
    include: 'relu',
    excludes: [
      'derivative',               // Not yet implemented.
      'gradient',                 // Not yet implemented.
      'valueAndGradients',        // Not yet implemented.
      'broadcasted bias',         // Not yet implemented.
      'fused A x B with 2d bias'  // Fused matMul with 2D bias not yet
                                  // supported.
    ]
  },
  {
    include: 'maxPool',
    excludes: [
      'ignores NaNs',      // Actual != expected.
    ]
  },
  {include: 'cropAndResize'},
  {include: 'resizeBilinear'},
  {include: 'resizeNearestNeighbor'},
  {
    include: 'matmul ',
    excludes: [
      'valueAndGradients',         // Gradients not defined yet
      'gradient',                  // Gradients not defined yet
      'zero in its shape',         // Zero in shapes aren't supported yet
      'matmul followed by mul',    // mul not supported yet
      'upcasts',                   // Upcasting not supported yet.
      'fused A x B with 2d bias',  // Fused matMul with 2D bias not yet
                                   // supported.
    ]
  },
  {
    include: 'depthwiseConv2D ',
    excludes: [
      'broadcasted bias',  // Broadcasted bias not yet supported.
      'gradient',          // Gradients not defined yet.
      'NCHW',              // xnn pack does not support channels first.
    ]
  },
  {
    include: 'conv2d ',
    excludes: [
      'broadcasted bias',  // Broadcasted bias not yet supported.
      'gradient',          // Gradients not defined yet.
      'backProp input x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // Gradients not
                                                         // defined.
      'NCHW',  // xnn pack does not support channels first.
      // Issue: https://github.com/tensorflow/tfjs/issues/3104.
      // Actual != expected.
      'relu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
      'prelu bias stride 2 x=[1,8,8,16] f=[3,3,16,1] s=[2,2] d=8 p=same',
    ]
  },
  {include: 'conv2dTranspose ', excludes: ['gradient']},
  {
    include: 'prelu ',
    excludes: [
      'gradient',   // Gradient is missing.
      'derivative'  // Missing gradient.
    ]
  },
  {
    include: ' cast ',
    excludes: [
      'complex',  // Complex numbers not yet implemented.
      'shallow slice an input that was cast'  // Slice is not implemented.
    ]
  },
  {
    include: 'gather',
    excludes: [
      'gradient'  // Not yet implemented.
    ]
  },
  {
    include: 'sigmoid ',
    excludes: [
      'sigmoidCrossEntropy',  // Not yet implemented.
      'gradient'              // Not yet implemented.
    ]
  },
  {include: 'scatterND '},
  {include: 'tensorScatterUpdate '},
  {
    include: 'abs ',
    excludes: [
      'gradient',  // Not yet implemented.
      'complex'    // Complex numbers not supported yet.
    ]
  },
  {
    include: 'sub ',
    excludes: [
      'complex',   // Complex numbers not yet implemented.
      'gradient',  // Not yet implemented.
      'upcasts',   // Upcasting not supported yet.
    ]
  },
  {
    include: 'mul ',
    excludes: [
      'complex',   // Complex numbers not yet supported.
      'gradient',  // Gradient not defined yet.
    ]
  },
  {
    include: 'div ',
    excludes: [
      'gradient',  // Gradient not defined yet.
      'upcasts',   // Cast not supported yet.
      'divNoNan'   // divNoNan not yet implemented.
    ]
  },
  {
    include: 'batchNorm',
    excludes: [
      'gradient'  // Gradient is missing.
    ]
  },
  {include: 'slice '},
  {include: 'stridedSlice '},
  {include: 'rotate '},
  {include: 'rotateWithOffset'},
  {include: 'flipLeftRight '},
  {include: 'square '},
  {include: 'squaredDifference'},
  {
    startsWith: 'min ',
    excludes: [
      '2D, axis=0',  // Permuted axes requires transpose, which is not yet
                     // implemented.
      'index corresponds to start of a non-initial window',  // argMin not yet
                                                             // implemented.,
      'gradient'  // Gradients not yet implemented
    ]
  },
  {
    startsWith: 'max ',
    excludes: [
      'gradient'  // Gradients not yet implemented
    ]
  },
  {
    include: 'concat',
    excludes: [
      'complex',  // Complex numbers not supported yet
      'gradient'  // Split is not yet implemented
    ]
  },
  {include: 'transpose', excludes: ['accepts complex64 input']},
  {include: 'oneHot'},
  {include: 'split'},
  {include: 'pad ', excludes: ['complex', 'zerosLike']},
  {
    include: 'clip',
    excludes: [
      'gradient',
      'propagates NaNs',  // clip delegates to XNNPACK which does not make
                          // guarantees about behavior of nans.
      'basic vec4'        // basic vec4 also includes nans.
    ]
  },
  {include: 'addN'},
  {include: 'nonMaxSuppression'},
  {include: 'argmax '},
  {include: 'argmin '},
  {include: 'exp '},
  {include: 'elu '},
  {include: 'unstack'},
  {
    include: 'minimum',
    excludes: [
      'gradient',                                 // Not yet implemented.
      'broadcasts 2x1 Tensor2D and 2x2 Tensor2D'  // Broadcasting along inner
                                                  // dims not supported yet.
    ]
  },
  {
    include: 'maximum',
    excludes: [
      'gradient'  // Not yet implemented.
    ]
  },
  {
    include: 'log ',
  },
  {startsWith: 'equal ', excludes: ['string']},
  {include: 'greater ', excludes: ['string']},
  {
    include: 'greaterEqual',
    excludes: [
      'gradient',  // Not yet implemented.
      'string'
    ]
  },
  {include: 'less ', excludes: ['string']},
  {
    include: 'lessEqual',
    excludes: [
      'gradient',  // Not yet implemented.
      'string'
    ]
  },
  {include: 'notEqual', excludes: ['string']},
  {
    include: 'mean ',
    excludes: [
      'axis=0',  // Reduction not supported along inner dimensions.
    ]
  },
  {startsWith: 'reverse'},
  {startsWith: 'sum '},
  {startsWith: 'cumprod'},
  {startsWith: 'cumsum'},
  {startsWith: 'logicalAnd '},
  {startsWith: 'logicalNot '},
  {startsWith: 'logicalOr '},
  {startsWith: 'logicalXor '},
  {
    startsWith: 'tile ',
    excludes: [
      'gradient',      // Gradient not yet implemented.
      'string tensor'  // String tensors not yet implemented.
    ]
  },
  {startsWith: 'erf'},
  {startsWith: 'sin '},
  {startsWith: 'sinh '},
  {
    startsWith: 'cos ',
    excludes: [
      'gradient',  // Gradient not yet implemented.
    ]
  },
  {
    startsWith: 'cosh',
    excludes: [
      'gradient'  // Gradient not yet implemented.
    ]
  },
  {
    startsWith: 'tan',
    excludes: ['gradient']  // Gradient not yet implemented.

  },
  {
    startsWith: 'tanh ',
    excludes: ['gradient']  // Gradient not yet implemented.
  },
  {
    startsWith: 'rsqrt ',
    excludes: ['gradient']  // Gradient not yet implemented.
  },
  {
    startsWith: 'sqrt ',
    excludes: ['gradient']  // Gradient not yet implemented.
  },
  {
    startsWith: 'where ',
    excludes: [
      '1D condition with higher rank a and b',  // Fill not yet implemented.
      'gradient'                                // Gradient not yet implemented.
    ]
  },
  {
    startsWith: 'zerosLike',
    // Complex numbers not supported yet.
    excludes: ['complex'],
  },
  {
    startsWith: 'onesLike',
    // Complex numbers not supported yet.
    excludes: ['complex'],
  },
  {include: 'prod'},
  {include: 'floor'},
  {include: 'floorDiv'},
  {include: 'topk'},
  {include: 'expandDims'},
  {include: 'stack'},
  {
    include: 'round',
    // Pool is not supported yet.
    excludes: ['pool'],
  },
  {include: 'step kernel'},
  {include: 'ceil'},
  {include: 'mirrorPad'},
  {
    startsWith: 'all',
    excludes: [
      'ignores NaNs'  // Doesn't yet ignore NaN
    ]
  },
  {
    startsWith: 'any',
    excludes: [
      'ignores NaNs'  // Doesn't yet ignore NaN
    ]
  },
  {include: 'image.transform'},
  {include: 'batchToSpaceND'},
  {include: 'spaceToBatchND'},
  {include: 'sparseFillEmptyRows'},
  {include: 'sparseReshape'},
  {include: 'sparseSegmentMean'},
  {include: 'sparseSegmentSum'},
  {startsWith: 'sparseToDense', excludes: ['string']},
  {include: 'stringNGrams'},
  {include: 'stringSplit'},
  {include: 'stringToHashBucketFast'},
  {include: 'reciprocal'},
  {include: 'isNaN'},
  {include: 'atan '},
  {include: 'acos '},
  {include: 'acosh '},
  {include: 'asin '},
  {include: 'asinh '},
  {include: 'diag '},
  {include: 'denseBincount '},
  {include: 'bitwiseAnd'},
  {include: 'broadcastArgs '},
  {include: 'searchSorted '},
  {include: 'avgPool3d '},
  {include: 'avgPool3dBackprop '},
  {include: 'upperBound '},
  {include: 'lowerBound '},
  {include: 'dilation2d '},
  {include: 'localResponseNormalization '},
  {include: 'log1p '},
  {include: 'atan2 '},
  {include: 'atanh '},
  {include: 'isInf '},
  {include: 'isFinite '},
  {include: 'sign '},
  {include: 'selu '},
  {include: 'softplus '},
  {include: 'linspace'},
  {include: 'bincount'},
  {include: 'expm1 '},
  {include: 'multinomial'},
  {include: 'unique'},
  {include: 'conv3d'},
  {include: 'mod '},
];

const customInclude = (testName: string) => {
  // Include all regular describe() tests.
  if (testName.indexOf('test-wasm') < 0) {
    return true;
  }

  // Include all of the wasm specific tests.
  if (testName.startsWith('wasm')) {
    return true;
  }

  return false;
};
setupTestFilters(TEST_FILTERS, customInclude);

beforeAll(setupCachedWasmPaths, 30_000);

// Import and run all the tests from core.
// tslint:disable-next-line:no-imports-from-dist
// tslint:disable-next-line:no-require-imports
require('@tensorflow/tfjs-core/dist/tests');
// Import and run wasm tests
// tslint:disable-next-line:no-require-imports
require('./tests');
