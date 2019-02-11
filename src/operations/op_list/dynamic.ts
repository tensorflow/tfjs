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
    'tfOpName': 'NonMaxSuppressionV2',
    'category': 'dynamic',
    'inputs': [
      {'start': 0, 'name': 'boxes', 'type': 'tensor'},
      {'start': 1, 'name': 'scores', 'type': 'tensor'},
      {'start': 2, 'name': 'maxOutputSize', 'type': 'number'},
      {'start': 3, 'name': 'iouThreshold', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'NonMaxSuppressionV3',
    'category': 'dynamic',
    'inputs': [
      {'start': 0, 'name': 'boxes', 'type': 'tensor'},
      {'start': 1, 'name': 'scores', 'type': 'tensor'},
      {'start': 2, 'name': 'maxOutputSize', 'type': 'number'},
      {'start': 3, 'name': 'iouThreshold', 'type': 'number'},
      {'start': 4, 'name': 'scoreThreshold', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Where',
    'category': 'dynamic',
    'inputs': [
      {'start': 0, 'name': 'condition', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'ListDiff',
    'category': 'dynamic',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'y', 'type': 'tensor'},
    ],
    'attrs': [{
      'tfName': 'T',
      'name': 'dtype',
      'type': 'dtype',
      'notSupported': true
    }]
  }
];
