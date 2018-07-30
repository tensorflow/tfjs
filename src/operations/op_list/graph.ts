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
    'tfOpName': 'PlaceholderWithDefault',
    'dlOpName': 'placeholder',
    'category': 'graph',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'default', 'type': 'tensor'},
      {'tfParamName': 'shape', 'dlParamName': 'shape', 'type': 'shape'},
      {'tfParamName': 'dtype', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {
    'tfOpName': 'Placeholder',
    'dlOpName': 'placeholder',
    'category': 'graph',
    'params': [
      {'tfParamName': 'shape', 'dlParamName': 'shape', 'type': 'shape'},
      {'tfParamName': 'dtype', 'dlParamName': 'dtype', 'type': 'dtype'}
    ]
  },
  {'tfOpName': 'Const', 'dlOpName': 'const', 'category': 'graph'}, {
    'tfOpName': 'Identity',
    'dlOpName': 'identity',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Snapshot',
    'dlOpName': 'snapshot',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Rank',
    'dlOpName': 'rank',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Size',
    'dlOpName': 'size',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Shape',
    'dlOpName': 'shape',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'Print',
    'dlOpName': 'print',
    'category': 'graph',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}, {
        'tfInputIndex': 1,
        'tfInputParamLength': 1,
        'dlParamName': 'data',
        'type': 'tensors'
      },
      {'tfParamName': 'message', 'dlParamName': 'message', 'type': 'string'}, {
        'tfParamName': 'first_n',
        'dlParamName': 'firstN',
        'type': 'number',
        'notSupprted': true
      },
      {
        'tfParamName': 'summarize',
        'dlParamName': 'summarize',
        'type': 'number',
        'defaultValue': 3
      }
    ]
  },
  {'tfOpName': 'NoOp', 'dlOpName': 'noop', 'category': 'graph', 'params': []}, {
    'tfOpName': 'StopGradient',
    'dlOpName': 'stopGradient',
    'category': 'graph',
    'params': [{'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'}]
  },
  {
    'tfOpName': 'FakeQuantWithMinMaxVars',
    'dlOpName': 'fakeQuantWithMinMaxVars',
    'category': 'graph',
    'params': [
      {'tfInputIndex': 0, 'dlParamName': 'x', 'type': 'tensor'},
      {'tfParamName': 'min', 'dlParamName': 'min', 'type': 'number'},
      {'tfParamName': 'max', 'dlParamName': 'max', 'type': 'number'}
    ]
  }
];
