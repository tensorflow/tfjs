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
    'tfOpName': 'Equal',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'NotEqual',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Greater',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'GreaterEqual',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Less',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'LessEqual',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'LogicalAnd',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'LogicalNot',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'LogicalOr',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'a', 'type': 'tensor'},
      {'start': 1, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Select',
    'category': 'logical',
    'inputs': [
      {'start': 0, 'name': 'condition', 'type': 'tensor'},
      {'start': 1, 'name': 'a', 'type': 'tensor'},
      {'start': 2, 'name': 'b', 'type': 'tensor'},
    ],
    'attrs': [{
      'tfName': 'T',
      'name': 'dtype',
      'type': 'dtype',
      'notSupported': true
    }]
  }
];
