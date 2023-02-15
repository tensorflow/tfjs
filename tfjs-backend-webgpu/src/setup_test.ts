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
  // skip specific test cases for supported kernels
  {
    startsWith: 'abs ',
    excludes: [
      'complex64',  // Kernel 'ComplexAbs' not registered.
    ]
  },
  {
    startsWith: 'conv3d ',
    excludes: [
      'gradient',  // Not yet implemented.
    ]
  },
  {
    startsWith: 'cumsum ',
    excludes: [
      'gradient',  // gradient function not found.
    ]
  },
  {
    startsWith: 'dilation2d ',
    excludes: [
      'gradient'  // gradient function not found.
    ]
  },
  {
    startsWith: 'exp ',
    excludes: [
      'int32',  // TODO: fix precision problem.
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
    startsWith: 'nonMaxSuppression ',
    excludes: [
      'NonMaxSuppressionPadded'  // NonMaxSuppressionV4 not yet implemented.
    ]
  },
  {
    startsWith: 'prod ',
    excludes: [
      'gradients',  // Not yet implemented
    ]
  },
  {
    startsWith: 'resizeBilinear ',
    excludes: [
      'gradients',  // Not yet implemented
    ]
  },
  {
    startsWith: 'resizeNearestNeighbor ',
    excludes: [
      'gradients',  // Not yet implemented
    ]
  },

  // exclude unsupported kernels and to be fixed cases
  {
    include: ' webgpu ',
    excludes: [
      // Not implemented kernel list.
      'avgPool3d ',           'avgPool3dBackprop ',
      'conv3dTranspose ',     'maxPool3d ',
      'maxPool3dBackprop ',   'raggedGather ',
      'raggedRange ',         'raggedTensorToTensor ',
      'method otsu',  // round
      'sparseFillEmptyRows ', 'sparseReshape ',
      'sparseSegmentMean ',   'sparseSegmentSum ',
      'stringSplit ',         'stringToHashBucketFast ',
      'tensorScatterUpdate ', 'unique ',
      'unsortedSegmentSum ',  'valueAndGradients ',
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
