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
    'tfOpName': 'Equal',
    'dlOpName': 'equal',
    'category': 'logical',
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
    'tfOpName': 'NotEqual',
    'dlOpName': 'notEqual',
    'category': 'logical',
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
    'tfOpName': 'Greater',
    'dlOpName': 'greater',
    'category': 'logical',
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
    'tfOpName': 'GreaterEqual',
    'dlOpName': 'greaterEqual',
    'category': 'logical',
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
    'tfOpName': 'Less',
    'dlOpName': 'less',
    'category': 'logical',
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
    'tfOpName': 'LessEqual',
    'dlOpName': 'lessEqual',
    'category': 'logical',
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
    'tfOpName': 'LogicalAnd',
    'dlOpName': 'logicalAnd',
    'category': 'logical',
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
    'tfOpName': 'LogicalNot',
    'dlOpName': 'logicalNot',
    'category': 'logical',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'a', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'LogicalOr',
    'dlOpName': 'logicalOr',
    'category': 'logical',
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
    'tfOpName': 'Select',
    'dlOpName': 'where',
    'category': 'logical',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'condition', 'type': 'tensor'},
      {'tfInputIndex': 1, 'dlParamName': 'a', 'type': 'tensor'},
      {'tfInputIndex': 2, 'dlParamName': 'b', 'type': 'tensor'}, {
        'tfParamName': 'T',
        'dlParamName': 'dtype',
        'type': 'dtype',
        'notSupported': true
      }
    ]
  }
];
