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

import {OpMapper} from '../types';

export const json: OpMapper[] = [
  {
    'tfOpName': 'SparseFillEmptyRows',
    'category': 'sparse',
    'inputs': [
      {'start': 0, 'name': 'indices', 'type': 'tensor'},
      {'start': 1, 'name': 'values', 'type': 'tensor'},
      {'start': 2, 'name': 'denseShape', 'type': 'tensor'},
      {'start': 3, 'name': 'defaultValue', 'type': 'tensor'},
    ]
  },
  {
    'tfOpName': 'SparseReshape',
    'category': 'sparse',
    'inputs': [
      {'start': 0, 'name': 'inputIndices', 'type': 'tensor'},
      {'start': 1, 'name': 'inputShape', 'type': 'tensor'},
      {'start': 2, 'name': 'newShape', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'SparseSegmentMean',
    'category': 'sparse',
    'inputs': [
      {'start': 0, 'name': 'data', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'tensor'},
      {'start': 2, 'name': 'segmentIds', 'type': 'tensor'},
    ]
  },
  {
    'tfOpName': 'SparseSegmentSum',
    'category': 'sparse',
    'inputs': [
      {'start': 0, 'name': 'data', 'type': 'tensor'},
      {'start': 1, 'name': 'indices', 'type': 'tensor'},
      {'start': 2, 'name': 'segmentIds', 'type': 'tensor'},
    ]
  }
];
