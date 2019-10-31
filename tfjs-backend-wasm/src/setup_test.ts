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
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

setTestEnvs([{name: 'test-wasm', backendName: 'wasm', isDataSync: true}]);

const env = jasmine.getEnv();
// Account for --grep flag passed to karma by saving the existing specFilter.
const grepFilter = env.specFilter;

interface TestFilter {
  include: string;
  excludes?: string[];
}

/** Tests that have these substrings in their name will be included. */
const INCLUDE_LIST: TestFilter[] = [
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
    ]
  },
  {
    include: 'batchNorm',
    excludes: [
      'gradient'  // Gradient is missing.
    ]
  },
  {include: 'slice '}, {include: 'square '}
];

/**
 * Filter method that returns boolean, if a given test should run or be
 * ignored based on its name. The exclude list has priority over the
 * include list. Thus, if a test matches both the exclude and the include
 * list, it will be exluded.
 */
env.specFilter = spec => {
  // Filter out tests if the --grep flag is passed.
  if (!grepFilter(spec)) {
    return false;
  }

  const name = spec.getFullName();

  // Include all regular describe() tests.
  if (name.indexOf('test-wasm') < 0) {
    return true;
  }

  // Include all of the wasm specific tests.
  if (name.startsWith('wasm')) {
    return true;
  }

  // Include a describeWithFlags() test from tfjs-core only if the test is
  // in the include list.
  for (let i = 0; i < INCLUDE_LIST.length; ++i) {
    const testFilter = INCLUDE_LIST[i];
    if (name.indexOf(testFilter.include) > -1) {
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

// Import and run all the tests from core.
// tslint:disable-next-line:no-imports-from-dist
import '@tensorflow/tfjs-core/dist/tests';
