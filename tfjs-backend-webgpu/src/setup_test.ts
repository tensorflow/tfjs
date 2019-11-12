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

// tslint:disable-next-line: no-imports-from-dist
import {setTestEnvs, setupTestFilters, TestFilter} from '@tensorflow/tfjs-core/dist/jasmine_util';

setTestEnvs([{
  name: 'test-webgpu',
  backendName: 'webgpu',
  flags: {'WEBGPU_CPU_FORWARD': false},
  isDataSync: false,
}]);

const TEST_FILTERS: TestFilter[] = [
  {
    include: 'less',
    excludes: [
      'upcasts when dtypes dont match',  // Actual != expected.
      'NaNs in',                         // Actual != expected.
      'broadcasting Tensor2D shapes',    // Actual != expected.
      'derivat',                         // logicalAnd not yet implemented.
    ]
  },
  {
    include: 'clip',
    excludes: [
      'derivat',   // logicalAnd not yet implemented.
      'gradient',  // logicalAnd not yet implemented.
    ]
  },
  {
    include: 'greater',
    excludes: [
      'upcasts when dtypes dont match',  // Actual != expected.
      'NaNs in',                         // Actual != expected.
      'broadcasting Tensor2D shapes',    // Actual != expected.
      'works with 0 sized tensors',      // Timeout.
      'gradient',                        // zerosLike not yet implemented.
    ]
  },
  {
    include: 'div',
    excludes: [
      'broadcast 2D + 1D',               // Actual != expected.
      'upcasts when dtypes dont match',  // Actual != expected.
      'gradient',                        // square, sum not yet implemented.
    ]
  },
  {
    include: 'depthwise',
    excludes: [
      'gradient',  // depthwiseConv2DDerInput not yet implemented.
      'fused',     // Not yet implemented.
    ]
  },
  {
    include: 'fromPixels',
    excludes: [
      'HTMLVideolement',  // Failed to execute 'getImageData' on
                          // 'CanvasRenderingContext2D': The source width is 0
    ]
  },
  {
    include: 'argmax',
    excludes: [
      '5D',                        // Rank 5 is not yet implemented.
      '6D',                        // Rank 5 is not yet implemented.
      'accepts tensor with bool',  // Actual != Expected.
      'gradient',                  // zerosLike not yet implemented.
    ]
  },
  {
    include: 'concat',
    excludes: [
      'complex',                           // No complex support yet.
      'concat a large number of tensors',  // Actual != Expected.
      'gradient',                          // split not yet implemented.
    ]
  },
  {
    include: 'transpose',
    excludes: [
      'oneHot',          // Not yet implemented.
      'fused',           // Not yet implemented.
      'batched matmul',  // Actual != expected, shape mismatch.
      'shape has ones',  // Actual != expected.
      '5D',              // Rank 5 is not yet implemented.
      '6D',              // Rank 5 is not yet implemented.
    ]
  },
  {
    include: 'relu',
    excludes: [
      'valueAndGradients',  // sum not yet implemented.
      'gradient',           // sum not yet implemented.
      'prelu',              // Not yet implemented.
      'fused',              // Not yet implemented.
      '5D',                 // Rank 5 is not yet implemented.
      '6D',                 // Rank 5 is not yet implemented.
      'propagates NaNs',    // Arrays differ.
    ]
  },
  {
    include: 'resizeBilinear',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {include: 'floor divide ', excludes: []},
  {
    include: 'maxPool',
    excludes: [
      'maxPoolBackprop',  // Not yet implemented.
      'maxPool3d',        // Not yet implemented.
    ]
  },
  {
    include: 'pool',
    excludes: [
      'avg x=[',                          // backend.avgPool not implemented.
      'max x=[4,3,1] f=[2,2] s=1 d=2',    // spaceToBatchND not yet implemented.
      'max x=[2,4,4,1] f=[2,2] s=1 d=2',  // spaceToBatchND not yet implemented.
      'poolBackprop',  // maxPoolBackprop not yet implemented.
    ]
  },
  {
    include: 'matmul',
    excludes: [
      'matmulBatch',                     // Shape mismatch.
      'fused matmul',                    // FusedMatmul not yet implemented.
      'gradient',                        // Various: sum not yet implemented.
      'has zero in its shape',           // Test times out.
      'valueAndGradients',               // backend.sum() not yet implemented.
      'upcasts when dtypes dont match',  // Missing cast().
      '^t',              // Shape mismatch for transposed matmul.
      'batched matmul',  // Actual != expected, shape mismatch.
    ]
  },
  {
    include: 'add ',
    excludes: [
      'complex',                         // No complex support yet.
      'upcasts when dtypes dont match',  // Missing cast().
      'accepts a tensor-like object',    // Timeout.
      'broadcast inner dim of b',        // Arrays differ.
      '6D',                              // Rank 6 is not yet implemented.
      'add tensors with 0 in shape',     // Timeout.
      'gradient',                        // sum not yet implemented.
    ]
  },
  {include: 'subtract ', excludes: []},
  {
    include: 'mul ',
    excludes: [
      'int32 * int32',  // Actual != Expected.
      'broadcast',      // Various: Actual != Expected, compile fails, etc.
      'gradient',       // Various: sum not yet implemented.
      'complex',        // No complex support yet.
      'upcasts when dtypes dont match',  // Actual != expected.
    ]
  },
  {
    include: 'conv2d',
    excludes: [
      'NCHW',             // Not yet implemented.
      'gradient',         // 'conv2dDerInput' not yet implemented
      'fused',            // Not yet implemented.
      'conv2dTranspose',  // DerInput is not Implemented.
    ]
  },
  {
    include: 'pad',
    excludes: [
      'RFFT',   // 'zerosLike' not yet implemented.
      'frame',  // Slice not yet implemented.
      'grad',   // 'depthwiseConv2DDerFilter' not yet implemented, slice not yet
                // implemented
    ]
  }
];

const customInclude = (testName: string) => {
  // Include regular describe() tests.
  if (testName.indexOf('test-webgpu') < 0) {
    return true;
  }

  // Include webgpu specific tests.
  if (testName.startsWith('webgpu')) {
    return true;
  }

  return false;
};

setupTestFilters(TEST_FILTERS, customInclude);

// Import and run all the tests from core.
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/tests';
