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

import {OpMapper} from '../types';

export const json: OpMapper[] = [
  {
    'tfOpName': '_FusedMatMul',
    'category': 'matrices',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
      {'start': 2, end: 0, 'name': 'args', 'type': 'tensors'},
    ],
    'attrs': [
      {'tfName': 'num_args', 'name': 'numArgs', 'type': 'number'}, {
        'tfName': 'fused_ops',
        'name': 'fusedOps',
        'type': 'string[]',
        'defaultValue': []
      },
      {
        'tfName': 'epsilon',
        'name': 'epsilon',
        'type': 'number',
        'defaultValue': 0.0001
      },
      {
        'tfName': 'transpose_a',
        'name': 'transposeA',
        'type': 'bool',
        'defaultValue': false
      },
      {
        'tfName': 'transpose_b',
        'name': 'transposeB',
        'type': 'bool',
        'defaultValue': false
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'MatMul',
    'category': 'matrices',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'transpose_a',
        'name': 'transposeA',
        'type': 'bool',
        'defaultValue': false
      },
      {
        'tfName': 'transpose_b',
        'name': 'transposeB',
        'type': 'bool',
        'defaultValue': false
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'BatchMatMul',
    'category': 'matrices',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'adj_x',
        'name': 'transposeA',
        'type': 'bool',
        'defaultValue': false
      },
      {
        'tfName': 'adj_y',
        'name': 'transposeB',
        'type': 'bool',
        'defaultValue': false
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'BatchMatMulV2',
    'category': 'matrices',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {
        'tfName': 'adj_x',
        'name': 'transposeA',
        'type': 'bool',
        'defaultValue': false
      },
      {
        'tfName': 'adj_y',
        'name': 'transposeB',
        'type': 'bool',
        'defaultValue': false
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Transpose',
    'category': 'matrices',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'perm', 'type': 'number[]'},
    ],
    'attrs': [{
      'tfName': 'T',
      'name': 'dtype',
      'type': 'dtype',
      'notSupported': true
    }]
  }
];
