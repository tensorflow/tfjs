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

import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

setTestEnvs([{
  name: 'test-webgpu',
  backendName: 'webgpu',
  flags: {},
  isDataSync: false,
}]);

const env = jasmine.getEnv();

/** Tests that have these substrings in their name will be included. */
const INCLUDE_LIST: string[] = [
  'matmul', 'add ', 'subtract ', 'mul ', 'conv2d', 'pad', 'pool', 'maxPool',
  'floor divide ', 'resizeBilinear', 'relu', 'transpose', 'concat', 'argmax',
  'fromPixels'
];
/** Tests that have these substrings in their name will be excluded. */
const EXCLUDE_LIST: string[] = [
  'conv to matmul',         // Shader compile fails.
  'should not leak',        // Missing backend.memory().
  'does not leak',          // Missing backend.memory().
  'matmulBatch',            // Shape mismatch.
  'gradient',               // Various: Shape mismatch, cast missing, etc.
  'has zero in its shape',  // Test times out.
  'batched matmul',         // Shape mismatch.
  'upcasts when dtypes dont match',  // Missing cast().
  '^t',                              // Shape mismatch for transposed matmul.
  'fused matmul',                    // FusedMatmul not yet implemented.
  'valueAndGradients',               // backend.sum() not yet implemented.
  'works when followed by',          // Shader compile fails.
  'works when preceded by',          // Shader compile fails.
  'complex',                         // No complex support yet.
  '5D',                              // Rank 5 is not yet implemented.
  '6D',                              // Rank 6 is not yet implemented.
  '3D+scalar',                       // Shader compile fails.
  'broadcast',  // Various: Actual != Expected, compile fails, etc.
  'accepts a tensor-like object',         // Shader compile fails.
  'add tensors with 0 in shape',          // Timeout.
  'c + A',                                // Shader compile fails.
  'int32 * int32',                        // Actual != Expected.
  'conv2dTranspose',                      // DerInput is not Implemented.
  'd=2',                                  // Dilation is not implemented.
  'pad1d test-webgpu {} grad',            // Needs backends.slice().
  'pad 4D arrays',                        // Actual != Expected.
  'tensor.toString',                      // readSync() is not available.
  'pad2d test-webgpu {} grad',            // Needs backend.slice().
  'avg x=[',                              // backend.avgPool not implemented.
  'preserves zero values',                // Shader compile fails.
  'relu test-webgpu {} propagates NaNs',  // Timeout.
  'relu test-webgpu {} sets negative values to 0',  // Shader compile fails.
  'relu test-webgpu {} does nothing to positive',   // Shader compile fail.
  'prelu',                                          // Not yet implemented.
  'oneHot test-webgpu {} Depth 2, transposed diagonal',  // Not yet implemented.
  'concat zero-sized tensors',                           // Timeout.
  'concat a large number of tensors',                    // Actual != Expected.
  'concat tensors with 0 in their shape',                // Timeout.
  'argmax test-webgpu {} accepts tensor with bool',      // Actual != Expected.
  'RFFT',                                                // Not yet implemented.
];

/**
 * Filter method that returns boolean, if a given test should run or be
 * ignored based on its name. The exclude list has priority over the include
 * list. Thus, if a test matches both the exclude and the include list, it
 * will be exluded.
 */
env.specFilter = spec => {
  const name = spec.getFullName();
  // Return false (skip the test) if the test is in the exclude list.
  for (let i = 0; i < EXCLUDE_LIST.length; ++i) {
    if (name.indexOf(EXCLUDE_LIST[i]) > -1) {
      return false;
    }
  }

  // Include all regular describe() tests.
  if (name.indexOf('test-webgpu') < 0) {
    return true;
  }

  // Include all of the webgpu specific tests.
  if (name.startsWith('webgpu')) {
    return true;
  }

  // Include a describeWithFlags() test from tfjs-core only if the test is in
  // the include list.
  for (let i = 0; i < INCLUDE_LIST.length; ++i) {
    if (name.indexOf(INCLUDE_LIST[i]) > -1) {
      return true;
    }
  }
  // Otherwise ignore the test.
  return false;
};

// Import and run all the tests from core.
import '@tensorflow/tfjs-core/dist/tests';
