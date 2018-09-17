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
    'tfOpName': 'ResizeBilinear',
    'dlOpName': 'resizeBilinear',
    'category': 'image',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'images', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'size', 'type': 'number[]'}, {
        'tfParamName': 'align_corners',
        'dlParamName': 'alignCorners',
        'type': 'bool'
      },
      {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'ResizeNearestNeighbor',
    'dlOpName': 'resizeNearestNeighbor',
    'category': 'image',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'images', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'size', 'type': 'number[]'}, {
        'tfParamName': 'align_corners',
        'dlParamName': 'alignCorners',
        'type': 'bool'
      },
      {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'CropAndResize',
    'dlOpName': 'cropAndResize',
    'category': 'image',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'image', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'boxes', 'type': 'tensor'},
      {'tfInputIndex': 2, 'dlParamName': 'boxInd', 'type': 'tensor'},
      {'tfInputIndex': 3, 'dlParamName': 'cropSize', 'type': 'number[]'},
      {'tfParamName': 'method', 'dlParamName': 'method', 'type': 'string'}, {
        'tfParamName': 'extrapolation_value',
        'dlParamName': 'extrapolationValue',
        'type': 'number'
      }
    ]
  }
];
