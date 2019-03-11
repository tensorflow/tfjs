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
    'tfOpName': 'Fill',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'shape', 'type': 'number[]'},
      {'start': 1, 'name': 'value', 'type': 'number'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'LinSpace',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'start', 'type': 'number'},
      {'start': 1, 'name': 'stop', 'type': 'number'},
      {'start': 2, 'name': 'num', 'type': 'number'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'OneHot',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'indices', 'type': 'tensor'},
      {'start': 1, 'name': 'depth', 'type': 'number'},
      {'start': 2, 'name': 'onValue', 'type': 'number', 'defaultValue': 1},
      {'start': 3, 'name': 'offValue', 'type': 'number', 'defaultValue': 0},
    ],
    'attrs': [
      {
        'tfName': 'axis',
        'name': 'axis',
        'type': 'number',
        'notSupported': true
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Ones',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'shape', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'OnesLike',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [{'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'RandomUniform',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'shape', 'type': 'number[]'},
    ],
    'attrs': [
      {
        'tfName': 'minval',
        'name': 'minval',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfName': 'maxval',
        'name': 'maxval',
        'type': 'number',
        'defaultValue': 1
      },
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'},
      {'tfName': 'seed', 'name': 'seed', 'type': 'number', 'defaultValue': 0}, {
        'tfName': 'seed2',
        'name': 'seed2',
        'type': 'number',
        'defaultValue': 0,
        'notSupported': true
      },
      {'tfName': 'T', 'name': 'T', 'type': 'number', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Range',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'start', 'type': 'number'},
      {'start': 1, 'name': 'stop', 'type': 'number'},
      {'start': 2, 'name': 'step', 'type': 'number', 'defaultValue': 0},
    ],
    'attrs': [{'tfName': 'Tidx', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'TruncatedNormal',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'shape', 'type': 'number[]'},
    ],
    'attrs': [
      {
        'tfName': 'means',
        'name': 'mean',
        'type': 'number',
        'defaultValue': 0.0
      },
      {
        'tfName': 'stddev',
        'name': 'stdDev',
        'type': 'number',
        'defaultValue': 1.0
      },
      {'tfName': 'seed', 'name': 'seed', 'type': 'number'}, {
        'tfName': 'seed2',
        'name': 'seed2',
        'type': 'number',
        'defaultValue': 0,
        'notSupported': true
      },
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'},
      {'tfName': 'T', 'name': 'T', 'type': 'number', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Zeros',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'shape', 'type': 'number[]'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  },
  {
    'tfOpName': 'ZerosLike',
    'category': 'creation',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [{'tfName': 'T', 'name': 'dtype', 'type': 'dtype'}]
  }
];
