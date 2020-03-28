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

import * as tensorflow from '../data/compiled_api';

import * as arithmetic from './op_list/arithmetic';
import * as basicMath from './op_list/basic_math';
import * as control from './op_list/control';
import * as convolution from './op_list/convolution';
import * as creation from './op_list/creation';
import * as dynamic from './op_list/dynamic';
import * as evaluation from './op_list/evaluation';
import * as graph from './op_list/graph';
import * as image from './op_list/image';
import * as logical from './op_list/logical';
import * as matrices from './op_list/matrices';
import * as normalization from './op_list/normalization';
import * as reduction from './op_list/reduction';
import * as sliceJoin from './op_list/slice_join';
import * as spectral from './op_list/spectral';
import * as transformation from './op_list/transformation';
import {OperationMapper} from './operation_mapper';
import {Graph} from './types';

const ops = [
  arithmetic, basicMath, control, convolution, creation, dynamic, evaluation,
  logical, image, graph, matrices, normalization, reduction, sliceJoin,
  spectral, transformation
];
const mapper: OperationMapper = OperationMapper.Instance;
let convertedGraph: Graph;

const SIMPLE_MODEL: tensorflow.IGraphDef = {
  node: [
    {
      name: 'image_placeholder',
      op: 'Placeholder',
      attr: {
        dtype: {
          type: tensorflow.DataType.DT_FLOAT,
        },
        shape:
            {shape: {dim: [{size: '3'}, {size: 3}, {size: '3'}, {size: 1}]}}
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
    {name: 'Fill', op: 'Fill', input: ['Shape', 'Value'], attr: {}},
    {
      name: 'Conv2D',
      op: 'Conv2D',
      input: ['image_placeholder', 'Const'],
      attr: {
        T: {type: tensorflow.DataType.DT_FLOAT},
        dataFormat: {s: 'TkhXQw=='},
        padding: {s: 'U0FNRQ=='},
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
        dataFormat: {s: 'TkhXQw=='}
      }
    },
    {
      name: 'Cast',
      op: 'Cast',
      input: ['BiasAdd'],
      attr: {DstT: {type: tensorflow.DataType.DT_INT64}}
    },
    {
      name: 'Squeeze',
      op: 'Squeeze',
      input: ['Cast'],
      attr: {squeeze_dims: {list: {i: ['1', '2']}}}
    },
    {
      name: 'Squeeze2',
      op: 'Squeeze',
      input: ['BiasAdd'],
      attr: {squeeze_dims: {list: {}}}
    },
    {
      name: 'Split',
      op: 'Split',
      input: ['image_placeholder'],
      attr: {num_split: {i: 3} as tensorflow.IAttrValue}
    },
    {name: 'LogicalNot', op: 'LogicalNot', input: ['image_placeholder']},
    {
      name: 'FusedBatchNorm',
      op: 'FusedBatchNorm',
      input: ['image_placeholder'],
      attr: {epsilon: {f: 0.0001} as tensorflow.IAttrValue}
    },
    {
      name: 'Cast2',
      op: 'Cast',
      input: ['BiasAdd'],
      attr: {DstT: {type: tensorflow.DataType.DT_UINT8}}
    },
  ],
  versions: {producer: 1.0}
};

const SIGNATURE: tensorflow.ISignatureDef = {
  inputs: {
    image: {
      name: 'image_placeholder',
      dtype: tensorflow.DataType.DT_INT32,
      tensorShape: {

      }
    }
  },
  outputs: {
    squeeze: {
      name: 'Squeeze',
      dtype: tensorflow.DataType.DT_FLOAT,
      tensorShape: {}
    }
  }
};

describe('completeness check', () => {
  it('should convert all op categories', () => {
    ops.forEach(op => {
      op.json.forEach(tfOp => {
        const graph = {
          node: [{name: tfOp.tfOpName, op: tfOp.tfOpName, attr: {}}]
        };
        convertedGraph = mapper.transformGraph(graph);
        expect(Object.keys(convertedGraph.nodes)).toEqual([tfOp.tfOpName]);
        expect(convertedGraph.nodes[tfOp.tfOpName].op).toEqual(tfOp.tfOpName);
      });
    });
  });
});
describe('operationMapper without signature', () => {
  beforeEach(() => {
    convertedGraph = mapper.transformGraph(SIMPLE_MODEL);
  });
  afterEach(() => {});

  describe('transform graph', () => {
    describe('graph level', () => {
      it('should find the graph input nodes', () => {
        expect(convertedGraph.inputs.map(node => node.name)).toEqual([
          'image_placeholder'
        ]);
      });

      it('should find the graph output nodes', () => {
        expect(convertedGraph.outputs.map(node => node.name)).toEqual([
          'Fill', 'Squeeze', 'Squeeze2', 'Split', 'LogicalNot',
          'FusedBatchNorm', 'Cast2'
        ]);
      });

      it('should find the graph weight nodes', () => {
        expect(convertedGraph.weights.map(node => node.name)).toEqual([
          'Const', 'Shape', 'Value'
        ]);
      });

      it('should convert nodes', () => {
        expect(Object.keys(convertedGraph.nodes)).toEqual([
          'image_placeholder', 'Const', 'Shape', 'Value', 'Fill', 'Conv2D',
          'BiasAdd', 'Cast', 'Squeeze', 'Squeeze2', 'Split', 'LogicalNot',
          'FusedBatchNorm', 'Cast2'
        ]);
      });
    });

    describe('node level', () => {
      it('should find the input nodes', () => {
        expect(convertedGraph.nodes['Fill'].inputs.map(node => node.name))
            .toEqual(['Shape', 'Value']);
      });
      it('should find the children nodes', () => {
        expect(convertedGraph.nodes['image_placeholder'].children.map(
                   node => node.name))
            .toEqual(['Conv2D', 'Split', 'LogicalNot', 'FusedBatchNorm']);
      });

      it('should map the input params', () => {
        expect(
            convertedGraph.nodes['Fill'].inputParams['shape'].inputIndexStart)
            .toEqual(0);
        expect(
            convertedGraph.nodes['Fill'].inputParams['value'].inputIndexStart)
            .toEqual(1);
      });

      it('should map the attribute params', () => {
        expect(convertedGraph.nodes['Conv2D'].attrParams['strides'].value)
            .toEqual([1, 2, 2, 1]);
        expect(convertedGraph.nodes['Conv2D'].attrParams['pad'].value)
            .toEqual('same');
        expect(convertedGraph.nodes['Conv2D'].attrParams['useCudnnOnGpu'].value)
            .toEqual(true);
        expect(
            convertedGraph.nodes['Split'].attrParams['numOrSizeSplits'].value)
            .toEqual(3);
        expect(
            convertedGraph.nodes['FusedBatchNorm'].attrParams['epsilon'].value)
            .toEqual(0.0001);
        expect(convertedGraph.nodes['Squeeze2'].attrParams['axis'].value)
            .toEqual([]);
      });

      it('should map the placeholder attribute params', () => {
        expect(
            convertedGraph.nodes['image_placeholder'].attrParams['shape'].value)
            .toEqual([3, 3, 3, 1]);
        expect(
            convertedGraph.nodes['image_placeholder'].attrParams['dtype'].value)
            .toEqual('float32');
      });
      it('should map params with deprecated name', () => {
        expect(convertedGraph.nodes['Squeeze'].attrParams['axis'].value)
            .toEqual([1, 2]);
      });
      it('should map params with int64 dtype', () => {
        expect(convertedGraph.nodes['Cast'].attrParams['dtype'].value)
            .toEqual('int32');
      });
    });
  });
});
describe('operationMapper with signature', () => {
  beforeEach(() => {
    convertedGraph = mapper.transformGraph(SIMPLE_MODEL, SIGNATURE);
  });
  afterEach(() => {});

  describe('transform graph', () => {
    describe('graph level', () => {
      it('should find the graph input nodes', () => {
        expect(convertedGraph.inputs.map(node => node.name)).toEqual([
          'image_placeholder'
        ]);
        expect(convertedGraph.inputs.map(node => node.signatureKey)).toEqual([
          'image'
        ]);
      });

      it('should find the graph output nodes', () => {
        expect(convertedGraph.outputs.map(node => node.name)).toEqual([
          'Squeeze'
        ]);
        expect(convertedGraph.outputs.map(node => node.signatureKey)).toEqual([
          'squeeze'
        ]);
      });

      it('should find the graph weight nodes', () => {
        expect(convertedGraph.weights.map(node => node.name)).toEqual([
          'Const', 'Shape', 'Value'
        ]);
      });

      it('should convert nodes', () => {
        expect(Object.keys(convertedGraph.nodes)).toEqual([
          'image_placeholder', 'Const', 'Shape', 'Value', 'Fill', 'Conv2D',
          'BiasAdd', 'Cast', 'Squeeze', 'Squeeze2', 'Split', 'LogicalNot',
          'FusedBatchNorm', 'Cast2'
        ]);
      });
    });

    describe('node level', () => {
      it('should find the input nodes', () => {
        expect(convertedGraph.nodes['Fill'].inputs.map(node => node.name))
            .toEqual(['Shape', 'Value']);
      });
      it('should find the children nodes', () => {
        expect(convertedGraph.nodes['image_placeholder'].children.map(
                   node => node.name))
            .toEqual(['Conv2D', 'Split', 'LogicalNot', 'FusedBatchNorm']);
      });

      it('should map the input params', () => {
        expect(
            convertedGraph.nodes['Fill'].inputParams['shape'].inputIndexStart)
            .toEqual(0);
        expect(
            convertedGraph.nodes['Fill'].inputParams['value'].inputIndexStart)
            .toEqual(1);
      });

      it('should map the attribute params', () => {
        expect(convertedGraph.nodes['Conv2D'].attrParams['strides'].value)
            .toEqual([1, 2, 2, 1]);
        expect(convertedGraph.nodes['Conv2D'].attrParams['pad'].value)
            .toEqual('same');
        expect(convertedGraph.nodes['Conv2D'].attrParams['useCudnnOnGpu'].value)
            .toEqual(true);
        expect(
            convertedGraph.nodes['Split'].attrParams['numOrSizeSplits'].value)
            .toEqual(3);
        expect(
            convertedGraph.nodes['FusedBatchNorm'].attrParams['epsilon'].value)
            .toEqual(0.0001);
        expect(convertedGraph.nodes['Squeeze2'].attrParams['axis'].value)
            .toEqual([]);
      });

      it('should map the placeholder attribute params', () => {
        expect(
            convertedGraph.nodes['image_placeholder'].attrParams['shape'].value)
            .toEqual([3, 3, 3, 1]);
        expect(
            convertedGraph.nodes['image_placeholder'].attrParams['dtype'].value)
            .toEqual('float32');
      });
      it('should map params with deprecated name', () => {
        expect(convertedGraph.nodes['Squeeze'].attrParams['axis'].value)
            .toEqual([1, 2]);
      });
      it('should map params with int64 dtype', () => {
        expect(convertedGraph.nodes['Cast'].attrParams['dtype'].value)
            .toEqual('int32');
      });
      it('should map params with uint8 dtype', () => {
        expect(convertedGraph.nodes['Cast2'].attrParams['dtype'].value)
            .toEqual('int32');
      });
    });
  });
});
