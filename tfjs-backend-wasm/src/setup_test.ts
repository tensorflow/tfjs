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
    include: 'add ',
    excludes: [
      'gradient',                   // Gradient is missing.
      'broadcast inner dim',        // Broadcast inner dim not yet supported.
      'broadcast each with 1 dim',  // Same as above.
      'broadcasting same rank Tensors different shape',  // Same as above.
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
    include: 'maxPool',
    excludes: [
      'f=[1,1]',          // XNN does not support filter height and width of 1.
      'maxPoolBackprop',  // Not yet implemented.
      'maxPool3d',        // Not yet implemented.
      'maxPool3dBackprop',  // Not yet implemented.
      'ignores NaNs'        // Actual != expected.
    ]
  },
  {include: 'cropAndResize'},
  {
    include: 'matmul ',
    excludes: [
      'valueAndGradients',       // Gradients not defined yet
      'gradient',                // Gradients not defined yet
      'fused matmul',            // Fused kernels aren't ready yet
      'zero in its shape',       // Zero in shapes aren't supported yet
      'matmul followed by mul',  // mul not supported yet
      'upcasts',                 // Upcasting not supported yet.
    ]
  },
  {
    include: 'depthwiseConv2D ',
    excludes: [
      'gradient',  // Gradients not defined yet.
      'NCHW',      // xnn pack does not support channels first.
    ]
  },
  {
    include: 'conv2d ',
    excludes: [
      // conv2d fusion is only done for bias.
      'im2row with bias and relu', 'im2row with prelu', 'basic with prelu',
      'basic with bias and relu', 'basic with elu', 'basic with relu',
      'pointwise with prelu', 'im2row with relu', 'im2row',
      'gradient',  // Gradients not defined yet.
      'NCHW',      // xnn pack does not support channels first.
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
    include: 'sigmoid ',
    excludes: [
      'sigmoidCrossEntropy',  // Not yet implemented.
      'gradient'              // Not yet implemented.
    ]
  },
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
      'complex',              // Complex numbers not yet implemented.
      'gradient',             // Not yet implemented.
      'upcasts',              // Upcasting not supported yet.
      'broadcast inner dim',  //  Broadcasting along inner dims not supported.
      'broadcast each with 1 dim',  //  Broadcasting along inner dims not
                                    //  supported.
      'broadcasting same rank Tensors different shape',  //  Broadcasting along
                                                         //  inner dims not
                                                         //  supported.
    ]
  },
  {
    include: 'mul ',
    excludes: [
      'complex',   // Complex numbers not yet supported.
      'gradient',  // Gradient not defined yet.
      'broadcasting same rank Tensors different shape',  // Broadcasting along
                                                         // inner dims not
                                                         // supported yet.
      'broadcast 5D + 2D',  // Broadcasting along inner dims not supported yet.
      'broadcast 6D + 2D'   // Broadcasting along inner dims not supported yet.
    ]
  },
  {
    include: 'div ',
    excludes: [
      'gradient',          // Gradient not defined yet.
      'integer division',  // FloorDiv not yet implemented.
      'upcasts',           // Cast not supported yet.
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
                                                             // implemented.
    ]
  },
  {
    startsWith: 'max ',
    excludes: [
      'derivative: 1D tensor with max or min value',  // Clip not yet
                                                      // implemented.
      '2D, axis=0'  // Permuted axes requires transpose, which is not yet
                    // implemented.
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
  {include: 'pad ', excludes: ['complex', 'zerosLike']},
  {include: 'clip', excludes: ['gradient']},
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
