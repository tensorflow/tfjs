/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
    'tfOpName': 'HashTable',
    'category': 'hash_table',
    'inputs': [],
    'attrs': [
      {'tfName': 'shared_name', 'name': 'sharedName', 'type': 'string'},
      {
        'tfName': 'use_node_name_sharing',
        'name': 'useNodeNameSharing',
        'type': 'bool'
      },
      {'tfName': 'key_dtype', 'name': 'keyDType', 'type': 'dtype'},
      {'tfName': 'value_dtype', 'name': 'valueDType', 'type': 'dtype'},
    ]
  },
  {
    'tfOpName': 'HashTableV2',
    'category': 'hash_table',
    'inputs': [],
    'attrs': [
      {'tfName': 'shared_name', 'name': 'sharedName', 'type': 'string'},
      {
        'tfName': 'use_node_name_sharing',
        'name': 'useNodeNameSharing',
        'type': 'bool'
      },
      {'tfName': 'key_dtype', 'name': 'keyDType', 'type': 'dtype'},
      {'tfName': 'value_dtype', 'name': 'valueDType', 'type': 'dtype'},
    ]
  },
  {
    'tfOpName': 'LookupTableImport',
    'category': 'hash_table',
    'inputs': [
      {'start': 0, 'name': 'tableHandle', 'type': 'tensor'},
      {'start': 1, 'name': 'keys', 'type': 'tensor'},
      {'start': 2, 'name': 'values', 'type': 'tensor'}
    ],
    'attrs': [
      {'tfName': 'Tin', 'name': 'tIn', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'tOut',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'LookupTableImportV2',
    'category': 'hash_table',
    'inputs': [
      {'start': 0, 'name': 'tableHandle', 'type': 'tensor'},
      {'start': 1, 'name': 'keys', 'type': 'tensor'},
      {'start': 2, 'name': 'values', 'type': 'tensor'}
    ],
    'attrs': [
      {'tfName': 'Tin', 'name': 'tIn', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'tOut',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'LookupTableFind',
    'category': 'hash_table',
    'inputs': [
      {'start': 0, 'name': 'tableHandle', 'type': 'tensor'},
      {'start': 1, 'name': 'keys', 'type': 'tensor'},
      {'start': 2, 'name': 'defaultValue', 'type': 'tensor'}
    ],
    'attrs': [
      {'tfName': 'Tin', 'name': 'tIn', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'tOut',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'LookupTableFindV2',
    'category': 'hash_table',
    'inputs': [
      {'start': 0, 'name': 'tableHandle', 'type': 'tensor'},
      {'start': 1, 'name': 'keys', 'type': 'tensor'},
      {'start': 2, 'name': 'defaultValue', 'type': 'tensor'}
    ],
    'attrs': [
      {'tfName': 'Tin', 'name': 'tIn', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'Tout',
        'name': 'tOut',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  }
];
