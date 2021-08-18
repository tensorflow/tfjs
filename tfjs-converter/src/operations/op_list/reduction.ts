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
    'tfOpName': 'Bincount',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'size', 'type': 'number'},
      {'start': 2, 'name': 'weights', 'type': 'tensor'}
    ]
  },
  {
    'tfOpName': 'DenseBincount',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'size', 'type': 'number'},
      {'start': 2, 'name': 'weights', 'type': 'tensor'}
    ],
    'attrs':
        [{'tfName': 'binary_output', 'name': 'binaryOutput', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Max',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Mean',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Min',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Sum',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'All',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Any',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'ArgMax',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'ArgMin',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Prod',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'keep_dims', 'name': 'keepDims', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Cumsum',
    'category': 'reduction',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'exclusive', 'name': 'exclusive', 'type': 'bool'},
      {'tfName': 'reverse', 'name': 'reverse', 'type': 'bool'}
    ]
  }
];
