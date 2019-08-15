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
    'tfOpName': 'ResizeBilinear',
    'category': 'image',
    'inputs': [
      {'start': 0, 'name': 'images', 'type': 'tensor'},
      {'start': 1, 'name': 'size', 'type': 'number[]'},
    ],
    'attrs': [
      {'tfName': 'align_corners', 'name': 'alignCorners', 'type': 'bool'},
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'ResizeNearestNeighbor',
    'category': 'image',
    'inputs': [
      {'start': 0, 'name': 'images', 'type': 'tensor'},
      {'start': 1, 'name': 'size', 'type': 'number[]'},
    ],
    'attrs': [
      {'tfName': 'align_corners', 'name': 'alignCorners', 'type': 'bool'},
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'CropAndResize',
    'category': 'image',
    'inputs': [
      {'start': 0, 'name': 'image', 'type': 'tensor'},
      {'start': 1, 'name': 'boxes', 'type': 'tensor'},
      {'start': 2, 'name': 'boxInd', 'type': 'tensor'},
      {'start': 3, 'name': 'cropSize', 'type': 'number[]'},
    ],
    'attrs': [
      {'tfName': 'method', 'name': 'method', 'type': 'string'}, {
        'tfName': 'extrapolation_value',
        'name': 'extrapolationValue',
        'type': 'number'
      }
    ]
  }
];
