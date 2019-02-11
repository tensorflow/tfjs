import {OpMapper} from '../types';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

export const json: OpMapper[] = [
  {
    'tfOpName': 'FusedBatchNorm',
    'category': 'normalization',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'scale', 'type': 'tensor'},
      {'start': 2, 'name': 'offset', 'type': 'tensor'},
      {'start': 3, 'name': 'mean', 'type': 'tensor'},
      {'start': 4, 'name': 'variance', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'epsilon',
        'name': 'epsilon',
        'type': 'number',
        'defaultValue': 0.001
      },
      {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'FusedBatchNormV2',
    'category': 'normalization',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'scale', 'type': 'tensor'},
      {'start': 2, 'name': 'offset', 'type': 'tensor'},
      {'start': 3, 'name': 'mean', 'type': 'tensor'},
      {'start': 4, 'name': 'variance', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'epsilon',
        'name': 'epsilon',
        'type': 'number',
        'defaultValue': 0.001
      },
      {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'LRN',
    'category': 'normalization',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'depth_radius',
        'name': 'radius',
        'type': 'number',
        'defaultValue': 5
      },
      {'tfName': 'bias', 'name': 'bias', 'type': 'number', 'defaultValue': 1.0},
      {
        'tfName': 'alpha',
        'name': 'alpha',
        'type': 'number',
        'defaultValue': 1.0
      },
      {
        'tfName': 'beta',
        'name': 'beta',
        'type': 'number',
        'defaultValue': 0.5
      }
    ]
  },
  {
    'tfOpName': 'Softmax',
    'category': 'normalization',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'LogSoftmax',
    'category': 'normalization',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'SparseToDense',
    'category': 'normalization',
    'inputs': [
      {'start': 0, 'name': 'sparseIndices', 'type': 'tensor'},
      {'start': 1, 'name': 'outputShape', 'type': 'number[]'},
      {'start': 2, 'name': 'sparseValues', 'type': 'tensor'},
      {'start': 3, 'name': 'defaultValue', 'type': 'tensor'},
    ],
    'attrs': [{
      'tfName': 'validate_indices',
      'name': 'validateIndices',
      'type': 'bool',
      'defaultValue': true,
      'notSupported': true
    }]
  }
];
