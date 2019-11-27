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
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

// TODO: Remove and import from tfjs-core once 1.3.2 is released, like so:
// import {setTestEnvs, setupTestFilters, TestFilter} from
// '@tensorflow/tfjs-core/dist/jasmine_util';

interface TestFilter {
  include?: string;
  startsWith?: string;
  excludes?: string[];
}

export function setupTestFilters(
    testFilters: TestFilter[], customInclude: (name: string) => boolean) {
  const env = jasmine.getEnv();
  // Account for --grep flag passed to karma by saving the existing specFilter.
  const grepFilter = env.specFilter;

  /**
   * Filter method that returns boolean, if a given test should run or be
   * ignored based on its name. The exclude list has priority over the
   * include list. Thus, if a test matches both the exclude and the include
   * list, it will be exluded.
   */
  // tslint:disable-next-line: no-any
  env.specFilter = (spec: any) => {
    // Filter out tests if the --grep flag is passed.
    if (!grepFilter(spec)) {
      return false;
    }

    const name = spec.getFullName();

    if (customInclude(name)) {
      return true;
    }

    // Include a describeWithFlags() test from tfjs-core only if the test is
    // in the include list.
    for (let i = 0; i < testFilters.length; ++i) {
      const testFilter = testFilters[i];
      if ((testFilter.include != null &&
           name.indexOf(testFilter.include) > -1) ||
          (testFilter.startsWith != null &&
           name.startsWith(testFilter.startsWith))) {
        if (testFilter.excludes != null) {
          for (let j = 0; j < testFilter.excludes.length; j++) {
            if (name.indexOf(testFilter.excludes[j]) > -1) {
              return false;
            }
          }
        }
        return true;
      }
    }
    // Otherwise ignore the test.
    return false;
  };
}

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
      'valueAndGradients',     // sum not yet implemented.
      'gradient',              // sum not yet implemented.
      'fused',                 // Not yet implemented.
      '5D',                    // Rank 5 is not yet implemented.
      '6D',                    // Rank 5 is not yet implemented.
      'propagates NaNs',       // Arrays differ.
      'derivative',            // sum not yet implemented.
      'gradient with clones',  // sum not yet implemented.
      'derivative where alpha got broadcasted',  // sum not yet implemented.
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
    include: 'fused',
    excludes: [
      'A x B',           // fusedBatchMatMul not yet implemented.
      'A x B with elu',  // elu not yet implemented.
      'A x B with elu and broadcasted bias',  // elu not yet implemented.
      'A x B with bias only',  // fusedBatchMatMul not yet implemented.
      'basic with elu',        // elu not yet implemented.
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0',  // conv2dDerInput not yet
                                                   // implemented.
      'gradient x=[2,3,3,1] f=[2,2,1,1] s=1 p=0 with bias',  // conv2dDerInput
                                                             // not yet
                                                             // implemented.
    ]
  },
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
      'batched matmul',                  // Actual != expected, shape mismatch.
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
    include: 'slice ',
    excludes: [
      'square a sliced texture',                 // abs not yet implemented.
      'square a non-sliced texture',             // abs not not yet implemented.
      'flatten a sliced tensor not continuous',  // square not yet implemented.
      'reshape a sliced 1d into a 2d tensor and',  // square not yet
                                                   // implemented.
      '5D',                  // Rank 5 is not yet implemented.
      '6D',                  // Rank 6 is not yet implemented.
      'strided slice with',  // Rank 6 is not yet implemented.
    ]
  },
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
