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
    'tfOpName': 'Fill',
    'dlOpName': 'fill',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'shape', 'type': 'number[]'},
      {'tfInputIndex': 1, 'dlParamName': 'value', 'type': 'number'},
      {'tfParamName': 'T', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'LinSpace',
    'dlOpName': 'linspace',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'start', 'type': 'number'},
      {'tfInputIndex': 1, 'dlParamName': 'stop', 'type': 'number'},
      {'tfInputIndex': 2, 'dlParamName': 'num', 'type': 'number'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'OneHot',
    'dlOpName': 'oneHot',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'indices', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'depth', 'type': 'number'}, {
        'tfInputIndex': 2,
        'dlParamName': 'onValue',
        'type': 'number',
        'defaultValue': 1
      },
      {
        'tfInputIndex': 3,
        'dlParamName': 'offValue',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'axis',
        'dlParamName': 'axis',
        'type': 'number',
        'notSupported': true
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
    'tfOpName': 'Ones',
    'dlOpName': 'ones',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'shape', 'type': 'number[]'},
      {'tfParamName': 'T', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'OnesLike',
    'dlOpName': 'onesLike',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfParamName': 'dtype', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'RandomUniform',
    'dlOpName': 'randomUniform',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'shape', 'type': 'number[]'}, {
        'tfParamName': 'minval',
        'dlParamName': 'minval',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'maxval',
        'dlParamName': 'maxval',
        'type': 'number',
        'defaultValue': 1
      },
      {'tfParamName': 'dtype', 'dlParamName': 'dtype', 'type': 'dtype'}, {
        'tfParamName': 'seed',
        'dlParamName': 'seed',
        'type': 'number',
        'defaultValue': 0
      },
      {
        'tfParamName': 'seed2',
        'dlParamName': 'seed2',
        'type': 'number',
        'defaultValue': 0,
        'notSupported': true
      },
      {
        'tfParamName': 'T',
        'dlParamName': 'T',
        'type': 'number',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Range',
    'dlOpName': 'range',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'start', 'type': 'number'},
      {'tfInputIndex': 1, 'dlParamName': 'stop', 'type': 'number'}, {
        'tfInputIndex': 2,
        'dlParamName': 'step',
        'type': 'number',
        'defaultValue': 0
      },
      {'tfParamName': 'Tidx', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'truncatedNormal',
    'dlOpName': 'truncatedNormal',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'shape', 'type': 'number[]'}, {
        'tfParamName': 'means',
        'dlParamName': 'mean',
        'type': 'number',
        'defaultValue': 0.0
      },
      {
        'tfParamName': 'stddev',
        'dlParamName': 'stdDev',
        'type': 'number',
        'defaultValue': 1.0
      },
      {'tfParamName': 'seed', 'dlParamName': 'seed', 'type': 'number'}, {
        'tfParamName': 'seed2',
        'dlParamName': 'seed2',
        'type': 'number',
        'defaultValue': 0,
        'notSupported': true
      },
      {'tfParamName': 'dtype', 'dlParamName': 'dtype', 'type': 'dtype'}, {
        'tfParamName': 'T',
        'dlParamName': 'T',
        'type': 'number',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Zeros',
    'dlOpName': 'zeros',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'shape', 'type': 'number[]'},
      {'tfParamName': 'T', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'ZerosLike',
    'dlOpName': 'zerosLike',
    'category': 'creation',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfParamName': 'T', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  }
];
