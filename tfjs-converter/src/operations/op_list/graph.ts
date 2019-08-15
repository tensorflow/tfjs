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
    'tfOpName': 'PlaceholderWithDefault',
    'category': 'graph',
    'inputs': [
      {'start': 0, 'name': 'default', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'shape', 'name': 'shape', 'type': 'shape'},
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'Placeholder',
    'category': 'graph',
    'attrs': [
      {'tfName': 'shape', 'name': 'shape', 'type': 'shape'},
      {'tfName': 'dtype', 'name': 'dtype', 'type': 'dtype'}
    ]
  },
  {'tfOpName': 'Const', 'category': 'graph'}, {
    'tfOpName': 'Identity',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'IdentityN',
    'category': 'graph',
    'inputs': [{'start': 0, 'end': 0, 'name': 'x', 'type': 'tensors'}]
  },
  {
    'tfOpName': 'Snapshot',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Rank',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Size',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Shape',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'ShapeN',
    'category': 'graph',
    'inputs': [{'start': 0, 'end': 0, 'name': 'x', 'type': 'tensors'}]
  },
  {
    'tfOpName': 'Print',
    'category': 'graph',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'data', 'type': 'tensors'},
    ],
    'attrs': [
      {'tfName': 'message', 'name': 'message', 'type': 'string'}, {
        'tfName': 'first_n',
        'name': 'firstN',
        'type': 'number',
        'notSupported': true
      },
      {
        'tfName': 'summarize',
        'name': 'summarize',
        'type': 'number',
        'defaultValue': 3
      }
    ]
  },
  {'tfOpName': 'NoOp', 'category': 'graph', 'inputs': []}, {
    'tfOpName': 'StopGradient',
    'category': 'graph',
    'inputs': [{'start': 0, 'name': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'FakeQuantWithMinMaxVars',
    'category': 'graph',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'min', 'name': 'min', 'type': 'number'},
      {'tfName': 'max', 'name': 'max', 'type': 'number'}
    ]
  }
];
