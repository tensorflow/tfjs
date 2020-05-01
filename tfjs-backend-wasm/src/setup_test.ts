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

// tslint:disable-next-line:no-imports-from-dist
import {setTestEnvs, setupTestFilters, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

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
      'gradient',  // zerosLike not defined yet.
      'broadcasting same rank Tensors different shape',  // Broadcasting along
                                                         // inner dims not
                                                         // supported yet.
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
  {
    include: 'avgPool',
    excludes: [
      'gradient',   // Not yet implemented.
      'avgPool3d',  // Not yet implemented.
    ]
  },
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
      'maxPoolBackprop',    // Not yet implemented.
      'maxPool3d',          // Not yet implemented.
      'maxPool3dBackprop',  // Not yet implemented.
      'ignores NaNs',       // Actual != expected.
      'maxPoolWithArgmax'   // Not yet implemented.

    ]
  },
  {include: 'cropAndResize'},
  {
    include: 'resizeBilinear',
    excludes: [
      'gradients'  // Not yet implemented.
    ]
  },
  {
    include: 'matmul ',
    excludes: [
      'valueAndGradients',         // Gradients not defined yet
      'gradient',                  // Gradients not defined yet
      'zero in its shape',         // Zero in shapes aren't supported yet
      'matmul followed by mul',    // mul not supported yet
      'upcasts',                   // Upcasting not supported yet.
      'fused A x B with elu',      // Fused matMul with elu activation not yet
                                   // supported.
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
      'basic with elu',    // Only fused relu, relu6, prelu activations
                           // supported.
      'gradient',          // Gradients not defined yet.
      'NCHW',              // xnn pack does not support channels first.
    ]
  },
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
      'broadcasting same rank Tensors different shape',  // Broadcasting along
                                                         // inner dims not
                                                         // supported yet.
      'divNoNan'  // divNoNan not yet implemented.
    ]
  },
  {
    include: 'batchNorm',
    excludes: [
      'gradient'  // Gradient is missing.
    ]
  },
  {include: 'slice '},
  {include: 'square '},
  {
    startsWith: 'min ',
    excludes: [
      'derivative: 1D tensor with max or min value',  // Clip not yet
                                                      // implemented.
      '2D, axis=0',  // Permuted axes requires transpose, which is not yet
                     // implemented.
      'index corresponds to start of a non-initial window',  // argMin not yet
                                                             // implemented.,
      'gradient',     // Gradients not yet implemented
      'ignores NaNs'  // Doesn't yet ignore NaN
    ]
  },
  {
    startsWith: 'max ',
    excludes: [
      'derivative: 1D tensor with max or min value',  // Clip not yet
                                                      // implemented.
      '2D, axis=0',   // Permuted axes requires transpose, which is not yet
                      // implemented.
      'gradient',     // Gradients not yet implemented
      'ignores NaNs'  // Doesn't yet ignore NaN
    ]
  },
  {
    include: 'concat',
    excludes: [
      'complex',  // Complex numbers not supported yet
      'gradient'  // Split is not yet implemented
    ]
  },
  {
    include: 'transpose',
    excludes: ['oneHot']  // oneHot not yet implemented.
  },
  {include: 'split'},
  {include: 'pad ', excludes: ['complex', 'zerosLike']},
  {include: 'clip', excludes: ['gradient']},
  {include: 'addN'},
  {include: 'nonMaxSuppression'},
  {include: 'argmax', excludes: ['gradient']},
  {include: 'exp '},
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
      'gradient',                                 // Not yet implemented.
      'broadcasts 2x1 Tensor2D and 2x2 Tensor2D'  // Broadcasting along inner
                                                  // dims not supported yet.
    ]
  },
  {
    include: 'log ',
  },
  {
    include: 'greater ',
    excludes: [
      'broadcasting Tensor2D shapes',  // Broadcasting along outer dims not
                                       // supported yet.
      'broadcasting Tensor3D shapes',  // Same as above.
      'broadcasting Tensor4D shapes'   // Same as above.
    ]
  },
  {
    include: 'greaterEqual',
    excludes: [
      'gradient',                      // Not yet implemented.
      'broadcasting Tensor2D shapes',  // Broadcasting along outer dims not
                                       // supported yet.
      'broadcasting Tensor3D shapes',  // Same as above.
      'broadcasting Tensor4D shapes'   // Same as above.
    ]
  },
  {
    include: 'less ',
    excludes: [
      'broadcasting Tensor2D shapes',   // Broadcasting along outer dims not
                                        // supported yet.
      'broadcasting Tensor3D shapes',   // Same as above.
      'broadcasting Tensor3D float32',  // Same as above.
      'broadcasting Tensor4D shapes'    // Same as above.
    ]
  },
  {
    include: 'lessEqual',
    excludes: [
      'gradient',                       // Not yet implemented.
      'broadcasting Tensor2D shapes',   // Broadcasting along outer dims not
                                        // supported yet.
      'broadcasting Tensor3D shapes',   // Same as above.
      'broadcasting Tensor3D float32',  // Same as above.
      'broadcasting Tensor4D shapes'    // Same as above.
    ]
  },
  {
    include: 'mean ',
    excludes: [
      'axis=0',  // Reduction not supported along inner dimensions.
    ]
  },
  {startsWith: 'sum '},
  {
    startsWith: 'logicalAnd ',
    excludes: [
      'broadcasting Tensor2D shapes',  // Broadcasting along outer dimensions
                                       // not yet supported.
      'broadcasting Tensor3D shapes',  // Same as above.
      'broadcasting Tensor4D shapes',  // Same as above.
    ]
  },
  {
    startsWith: 'tile ',
    excludes: [
      'gradient',      // Gradient not yet implemented.
      'string tensor'  // String tensors not yet implemented.
    ]
  },
  {startsWith: 'sin '},
  {
    startsWith: 'cos ',
    excludes: [
      'gradient',  // Gradient not yet implemented.
    ]
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
    startsWith: 'zerosLike',
    // Complex numbers not supported yet.
    excludes: ['complex'],
  },
  {
    startsWith: 'onesLike',
    // Complex numbers not supported yet.
    excludes: ['complex'],
  },
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

// Import and run all the tests from core.
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/tests';
