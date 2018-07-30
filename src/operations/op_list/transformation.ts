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
    'tfOpName': 'Cast',
    'dlOpName': 'cast',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}, {
        'tfParamName': 'SrcT',
        'dlParamName': 'sdtype',
        'type': 'dtype',
        'notSupported': true
      },
      {'tfParamName': 'DstT', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'ExpandDims',
    'dlOpName': 'expandDims',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}, {
        'tfInputIndex': 1,
        'tfParamNameDeprecated': 'dim',
        'dlParamName': 'axis',
        'type': 'number'
      }
    ]
  },
  {
    'tfOpName': 'Pad',
    'dlOpName': 'pad',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'padding', 'type': 'number[]'}, {
        'tfParamName': 'constant_value',
        'dlParamName': 'constantValue',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  },
  {
    'tfOpName': 'PadV2',
    'dlOpName': 'pad',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'padding', 'type': 'number[]'}, {
        'tfInputIndex': 2,
        'dlParamName': 'constantValue',
        'type': 'number',
        'defaultValue': 0
      }
    ]
  },
  {
    'tfOpName': 'Reshape',
    'dlOpName': 'reshape',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'shape', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'Squeeze',
    'dlOpName': 'squeeze',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}, {
        'tfParamName': 'axis',
        'tfParamNameDeprecated': 'squeeze_dims',
        'dlParamName': 'axis',
        'type': 'number[]'
      }
    ]
  },
  {
    'tfOpName': 'SpaceToBatchND',
    'dlOpName': 'spaceToBatchND',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'blockShape', 'type': 'number[]'},
      {'tfInputIndex': 2, 'dlParamName': 'paddings', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'BatchToSpaceND',
    'dlOpName': 'batchToSpaceND',
    'category': 'transformation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'blockShape', 'type': 'number[]'},
      {'tfInputIndex': 2, 'dlParamName': 'crops', 'type': 'number[]'}
    ]
  }
];
