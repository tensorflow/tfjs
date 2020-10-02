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
    'tfOpName': 'TopKV2',
    'category': 'evaluation',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'k', 'type': 'number'},
    ],
    'attrs': [{'tfName': 'sorted', 'name': 'sorted', 'type': 'bool'}]
  },
  {
    'tfOpName': 'Unique',
    'category': 'evaluation',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
  },
  {
    'tfOpName': 'UniqueV2',
    'category': 'evaluation',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'axis', 'type': 'number'},
    ],
  },
];
