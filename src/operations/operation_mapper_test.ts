/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {tensorflow} from '../data/index';

import {OperationMapper} from './index';

const mapper: OperationMapper = OperationMapper.Instance;
const SIMPLE_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'image_placeholder',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_FLOAT,
        },
        shape: {shape: {dim: [{size: 3}, {size: 3}, {size: 3}, {size: 1}]}}
      }
    },
    {
      name: 'Const',
      op: 'Const',
      attr: {
        dtype: {type: tensorflow.DataType.DT_INT32},
        value: {
          tensor: {
            dtype: tensorflow.DataType.DT_INT32,
            tensorShape: {dim: [{size: 3}, {size: 3}, {size: 1}, {size: 1}]},
            intVal: [0, 0, 0, 0, 1, 0, 0, 0, 0]
          }
        }
      }
    },
    {
      name: 'Shape',
      op: 'Const',
      attr: {
        dtype: {type: tensorflow.DataType.DT_INT32},
        value: {
          tensor: {
            dtype: tensorflow.DataType.DT_INT32,
            tensorShape: {dim: [{size: 3}, {size: 1}, {size: 1}, {size: 1}]},
            intVal: [1, 1, 1]
          }
        }
      }
    },
    {
      name: 'Conv2D',
      op: 'Conv2D',
      input: ['image_placeholder', 'Const'],
      attr: {
        T: {type: tensorflow.DataType.DT_FLOAT},
        dataFormat: {s: Uint8Array.from([, 12, 2])},
        padding: {s: Uint8Array.from([118, 97, 108, 105, 100])},
        strides: {list: {f: [], i: [1, 2, 2, 1]}},
        useCudnnOnGpu: {b: true}
      }
    },
    {
      name: 'BiasAdd',
      op: 'BiasAdd',
      input: ['Conv2D', 'Shape'],
      attr: {
        T: {type: tensorflow.DataType.DT_FLOAT},
        dataFormat: {s: Uint8Array.from([1, 2, 34])}
      }
    }
  ],
  versions: {producer: 1.0}
};

describe('operationMapper', () => {
  beforeEach(() => {});
  afterEach(() => {});

  it('should transform graph', () => {
    console.log(mapper.transformGraph(SIMPLE_MODEL));
  });
});
