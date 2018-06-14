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

import * as Long from 'long';

import {tensorflow} from '../data/compiled_api';
import {OperationMapper} from './operation_mapper';
import {Graph} from './types';

const mapper: OperationMapper = OperationMapper.Instance;
let graph: Graph;
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
      name: 'Value',
      op: 'Const',
      attr: {dtype: {type: tensorflow.DataType.DT_INT32}, value: {i: 1}}
    },
    {name: 'Fill', op: 'Fill', input: ['Shape', 'Value'], attr: {}}, {
      name: 'Conv2D',
      op: 'Conv2D',
      input: ['image_placeholder', 'Const'],
      attr: {
        T: {type: tensorflow.DataType.DT_FLOAT},
        dataFormat: {s: Uint8Array.from([1, 12, 2])},
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
    },
    {
      name: 'Squeeze',
      op: 'Squeeze',
      input: ['BiasAdd'],
      attr: {squeeze_dims: {list: {i: [Long.fromInt(1), Long.fromInt(2)]}}}
    }
  ],
  versions: {producer: 1.0}
};

describe('operationMapper', () => {
  beforeEach(() => {
    graph = mapper.transformGraph(SIMPLE_MODEL);
  });
  afterEach(() => {});

  describe('transform graph', () => {
    describe('graph level', () => {
      it('should find the graph input nodes', () => {
        expect(graph.inputs.map(node => node.name)).toEqual([
          'image_placeholder', 'Const', 'Shape', 'Value'
        ]);
      });

      it('should find the graph output nodes', () => {
        expect(graph.outputs.map(node => node.name)).toEqual([
          'Fill', 'Squeeze'
        ]);
      });

      it('should convert nodes', () => {
        expect(Object.keys(graph.nodes)).toEqual([
          'image_placeholder', 'Const', 'Shape', 'Value', 'Fill', 'Conv2D',
          'BiasAdd', 'Squeeze'
        ]);
      });
    });

    describe('node level', () => {
      it('should find the input nodes', () => {
        expect(graph.nodes['Fill'].inputs.map(node => node.name)).toEqual([
          'Shape', 'Value'
        ]);
      });
      it('should find the children nodes', () => {
        expect(graph.nodes['image_placeholder'].children.map(node => node.name))
            .toEqual(['Conv2D']);
      });

      it('should map the input params', () => {
        expect(graph.nodes['Fill'].params['shape'].inputIndex).toEqual(0);
        expect(graph.nodes['Fill'].params['value'].inputIndex).toEqual(1);
      });

      it('should map the attribute params', () => {
        expect(graph.nodes['Conv2D'].params['strides'].value).toEqual([
          1, 2, 2, 1
        ]);
        expect(graph.nodes['Conv2D'].params['pad'].value).toEqual('valid');
        expect(graph.nodes['Conv2D'].params['useCudnnOnGpu'].value)
            .toEqual(true);
      });

      it('should map the placeholder attribute params', () => {
        expect(graph.nodes['image_placeholder'].params['shape'].value).toEqual([
          3, 3, 3, 1
        ]);
        expect(graph.nodes['image_placeholder'].params['dtype'].value)
            .toEqual('float32');
      });
      it('should map params with deprecated name', () => {
        expect(graph.nodes['Squeeze'].params['axis'].value).toEqual([1, 2]);
      });
    });
  });
});
