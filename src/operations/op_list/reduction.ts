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

export const json = [
  {
    'tfOpName': 'Max',
    'dlOpName': 'max',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'Mean',
    'dlOpName': 'mean',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'Min',
    'dlOpName': 'min',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'Sum',
    'dlOpName': 'sum',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'All',
    'dlOpName': 'all',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'Any',
    'dlOpName': 'any',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'},
      {'tfParamName': 'keep_dims', 'dlParamName': 'keepDims', 'type': 'bool'}
    ]
  },
  {
    'tfOpName': 'ArgMax',
    'dlOpName': 'argMax',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'ArgMin',
    'dlOpName': 'argMin',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Prod',
    'dlOpName': 'prod',
    'category': 'reduction',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'axis', 'type': 'number[]'}, {
        'tfParamName': 'keep_dims',
        'dlParamName': 'keepDims',
        'type': 'bool'
      }
    ]
  }
];
