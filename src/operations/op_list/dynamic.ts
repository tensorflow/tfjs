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
    'tfOpName': 'NonMaxSuppressionV2',
    'dlOpName': 'nonMaxSuppression',
    'category': 'dynamic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'boxes', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'scores', 'type': 'tensor'},
      {'tfInputIndex': 2, 'dlParamName': 'maxOutputSize', 'type': 'number'},
      {'tfInputIndex': 3, 'dlParamName': 'iouThreshold', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'NonMaxSuppressionV3',
    'dlOpName': 'nonMaxSuppression',
    'category': 'dynamic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'boxes', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'scores', 'type': 'tensor'},
      {'tfInputIndex': 2, 'dlParamName': 'maxOutputSize', 'type': 'number'},
      {'tfInputIndex': 3, 'dlParamName': 'iouThreshold', 'type': 'number'},
      {'tfInputIndex': 4, 'dlParamName': 'scoreThreshold', 'type': 'number'}
    ]
  },
  {
    'tfOpName': 'Where',
    'dlOpName': 'whereAsync',
    'category': 'dynamic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'condition', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'ListDiff',
    'dlOpName': 'setdiff1dAsync',
    'category': 'dynamic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'y', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  }
];
