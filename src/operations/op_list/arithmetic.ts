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
    'tfOpName': 'Add',
    'dlOpName': 'add',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'AddN',
    'dlOpName': 'addN',
    'category': 'arithmetic',
    'params': [{
      'tfInputIndex': 0,
      'tfInputParamLength': 0,
      'dlParamName': 'tensors',
      'type': 'tensors'
    }]
  },
  {
    'tfOpName': 'BiasAdd',
    'dlOpName': 'add',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Sub',
    'dlOpName': 'sub',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'RealDiv',
    'dlOpName': 'div',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Div',
    'dlOpName': 'div',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'FloorDiv',
    'dlOpName': 'floorDiv',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Mul',
    'dlOpName': 'mul',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Maximum',
    'dlOpName': 'maximum',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}
    ]
  },
  {
    'tfOpName': 'Minimum',
    'dlOpName': 'minimum',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}
    ]
  },
  {
    'tfOpName': 'Pow',
    'dlOpName': 'pow',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'SquaredDifference',
    'dlOpName': 'squaredDifference',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'Mod',
    'dlOpName': 'mod',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'FloorMod',
    'dlOpName': 'mod',
    'category': 'arithmetic',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  }
];
