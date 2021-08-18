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
import * as tfc from '@tensorflow/tfjs-core';

import * as tensorflow from '../data/compiled_api';
import {createTensorAttr} from '../operations/executors/test_helper';
import {Graph, Node} from '../operations/types';

import {GraphExecutor} from './graph_executor';

let executor: GraphExecutor;
let inputNode: Node;
let constNode: Node;
let intermediateNode: Node;
let rsqrtNode: Node;
let outputNode: Node;
let graph: Graph;
let graphWithControlFlow: Graph;
let constTensor: tfc.Tensor;

const SIGNATURE: tensorflow.ISignatureDef = {
  inputs: {
    x: {name: 'input', dtype: tensorflow.DataType.DT_INT32, tensorShape: {}}
  },
  outputs: {
    add: {
      name: 'output',
      dtype: tensorflow.DataType.DT_FLOAT,
      tensorShape: {}
    }
  }
};

describe('GraphExecutor', () => {
  beforeEach(() => {
    inputNode = {
      inputNames: [],
      inputs: [],
      children: [],
      signatureKey: 'x',
      name: 'input',
      op: 'Placeholder',
      category: 'graph',
      attrParams: {},
      inputParams: {}
    };
    constNode = {
      inputNames: [],
      inputs: [],
      children: [],
      name: 'const',
      op: 'Const',
      category: 'graph',
      attrParams: {},
      inputParams: {}

    };
    intermediateNode = {
      inputNames: ['input', 'const'],
      inputs: [inputNode, constNode],
      children: [],
      name: 'intermediate',
      op: 'Add',
      category: 'arithmetic',
      inputParams: {'a': createTensorAttr(0), 'b': createTensorAttr(1)},
      attrParams: {}
    };
    outputNode = {
      inputNames: ['intermediate', 'const'],
      inputs: [intermediateNode, constNode],
      children: [],
      name: 'output',
      signatureKey: 'add',
      op: 'Add',
      category: 'arithmetic',
      inputParams: {'a': createTensorAttr(0), 'b': createTensorAttr(1)},
      attrParams: {}
    };
    graph = {
      inputs: [inputNode],
      nodes: {
        'input': inputNode,
        'const': constNode,
        'intermediate': intermediateNode,
        'output': outputNode
      },
      outputs: [outputNode],
      weights: [constNode],
      placeholders: [inputNode],
      functions: {
        while_body: {
          inputs: [inputNode],
          nodes: {
            'input': inputNode,
            'const': constNode,
            'intermediate': intermediateNode,
            'output': outputNode
          },
          outputs: [outputNode],
          weights: [constNode],
          placeholders: [inputNode],
          signature: SIGNATURE
        }
      },
      signature: SIGNATURE
    };
    inputNode.children.push(intermediateNode);
    constNode.children.push(intermediateNode, outputNode);
    intermediateNode.children.push(outputNode);
    executor = new GraphExecutor(graph);
    constTensor = tfc.scalar(2.0);
    executor.weightMap = {'const': [constTensor]};
  });
  afterEach(() => {});

  describe('execute graph', () => {
    describe('initialization', () => {
      it('should expose input names', () => {
        expect(executor.inputNodes).toEqual(['x']);
      });

      it('should expose output names', () => {
        expect(executor.outputNodes).toEqual(['add']);
      });

      it('should expose inputs', () => {
        inputNode.attrParams['shape'] = {value: [1], type: 'shape'};
        inputNode.attrParams['dtype'] = {value: 'float32', type: 'dtype'};
        expect(executor.inputs).toEqual([
          {name: 'input', shape: [1], dtype: 'float32'}
        ]);
      });

      it('should expose outputs', () => {
        outputNode.attrParams['shape'] = {value: [1, 1], type: 'shape'};
        outputNode.attrParams['dtype'] = {value: 'int32', type: 'dtype'};
        expect(executor.outputs).toEqual([
          {name: 'output', shape: [1, 1], dtype: 'int32'}
        ]);
      });

      it('should expose functions', () => {
        expect(executor.functions).toEqual({while_body: SIGNATURE});
      });
    });

    describe('graph level', () => {
      describe('execute', () => {
        it('should execute the op', async () => {
          const inputTensor = tfc.scalar(1);
          const result = executor.execute({input: inputTensor}, ['output']);
          tfc.test_util.expectArraysClose(await result[0].data(), [5.0]);
        });

        it('should allow output intermediate nodes', async () => {
          const inputTensor = tfc.scalar(1);
          const result = executor.execute(
              {input: inputTensor}, ['output', 'intermediate']);
          tfc.test_util.expectArraysClose(await result[1].data(), [3.0]);
          tfc.test_util.expectArraysClose(await result[0].data(), [5.0]);
        });

        it('should allow feed intermediate nodes', async () => {
          const intermediateTensor = tfc.scalar(1);
          const result =
              executor.execute({intermediate: intermediateTensor}, ['output']);
          tfc.test_util.expectArraysClose(await result[0].data(), [3.0]);
        });

        describe('strict input check', () => {
          it('should throw exception if missing inputs', () => {
            expect(() => executor.execute({}, ['output']))
                .toThrowError(
                    'Cannot compute the outputs [output] from the provided ' +
                    'inputs []. Missing the following inputs: [input]');
          });

          it('should throw exception if contains extra inputs', () => {
            const inputTensor = tfc.scalar(1);
            expect(
                () => executor.execute(
                    {test: inputTensor, input: inputTensor}, ['output']))
                .toThrowError(
                    'The dict provided in model.execute(dict) has keys: ' +
                    '[test] that are not part of graph');
          });

          it('should throw exception if inputs shapes mismatch', () => {
            inputNode.attrParams['shape'] = {value: [1, 1], type: 'shape'};
            const inputTensor = tfc.tensor1d([1], 'float32');
            expect(() => executor.execute({input: inputTensor}, ['output']))
                .toThrow(new Error(
                    'The shape of dict[\'input\'] provided' +
                    ' in model.execute(dict) must be [1,1], but was [1]'));
          });

          it('should throw exception for dtype mismatch', () => {
            inputNode.attrParams['dtype'] = {value: 'int32', type: 'dtype'};
            const inputTensor = tfc.tensor1d([1], 'float32');
            expect(() => executor.execute({input: inputTensor}, ['output']))
                .toThrow(new Error(
                    'The dtype of dict[\'input\'] provided' +
                    ' in model.execute(dict) must be int32, but was float32'));
          });
        });

        describe('outputs check', () => {
          it('should reject missing outputs', () => {
            const inputTensor = tfc.tensor1d([1], 'float32');
            expect(() => executor.execute({input: inputTensor}, ['missing']))
                .toThrowError(/The output 'missing' is not found in the graph/);
          });

          it('should reject missing outputs with child tensors', () => {
            const inputTensor = tfc.tensor1d([1], 'float32');
            expect(() => executor.execute({input: inputTensor}, ['missing:0']))
                .toThrowError(
                    /The output 'missing:0' is not found in the graph/);
          });

          it('should accept existing outputs', () => {
            const inputTensor = tfc.tensor1d([1], 'float32');
            const res = executor.execute({input: inputTensor}, ['output']);
            expect(res).not.toBeNull();
          });

          it('should accept existing outputs with child tensors', () => {
            const inputTensor = tfc.tensor1d([1], 'float32');
            const res = executor.execute({input: inputTensor}, ['output:0']);
            expect(res).not.toBeNull();
          });
        });

        it('should not throw exception if inputs shapes is dynamic', () => {
          inputNode.attrParams['shape'] = {value: [-1, 1, 1, 1], type: 'shape'};
          const inputTensor = tfc.tensor4d([1, 1], [2, 1, 1, 1], 'float32');
          const res = executor.execute({input: inputTensor}, ['output']);
          expect(res).not.toBeNull();
        });

        it('should not have mem leak when add index', async () => {
          const inputTensor = tfc.tensor4d([1, 1], [2, 1, 1, 1], 'float32');
          const numTensors: number = tfc.memory().numTensors;

          const res = executor.execute({input: inputTensor}, ['output:0']);
          expect(res).not.toBeNull();
          expect(tfc.memory().numTensors).toEqual(numTensors + 1);
        });
      });

      describe('executeAsync', () => {
        beforeEach(() => {
          inputNode = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'input',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          constNode = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'const',
            op: 'Const',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          intermediateNode = {
            inputNames: ['input', 'const'],
            inputs: [inputNode, constNode],
            children: [],
            name: 'intermediate',
            op: 'Add',
            category: 'arithmetic',
            inputParams: {'a': createTensorAttr(0), 'b': createTensorAttr(1)},
            attrParams: {}
          };
          rsqrtNode = {
            inputNames: ['intermediate'],
            inputs: [intermediateNode],
            children: [],
            name: 'rsqrt',
            op: 'Rsqrt',
            category: 'basic_math',
            inputParams: {'x': createTensorAttr(0)},
            attrParams: {}
          };
          outputNode = {
            inputNames: ['const', 'rsqrt'],
            inputs: [constNode, rsqrtNode],
            children: [],
            name: 'output',
            op: 'Switch',
            category: 'control',
            inputParams:
                {'pred': createTensorAttr(0), 'data': createTensorAttr(1)},
            attrParams: {}
          };
          inputNode.children.push(intermediateNode);
          constNode.children.push(intermediateNode, outputNode);
          intermediateNode.children.push(rsqrtNode);
          rsqrtNode.children.push(outputNode);
          graphWithControlFlow = {
            inputs: [constNode, inputNode],
            nodes: {
              'input': inputNode,
              'const': constNode,
              'intermediate': intermediateNode,
              'rsqrt': rsqrtNode,
              'output': outputNode
            },
            outputs: [outputNode],
            weights: [constNode],
            placeholders: [inputNode]
          };

          executor = new GraphExecutor(graphWithControlFlow);
          executor.weightMap = {const : [constTensor]};
        });

        it('should execute control flow graph', async () => {
          const inputTensor = tfc.scalar(1);

          const result =
              await executor.executeAsync({input: inputTensor}, ['output:1']);
          tfc.test_util.expectArraysClose(await result[0].data(), [0.57735]);
        });

        it('should allow output intermediate nodes', async () => {
          const inputTensor = tfc.scalar(1);
          const result = await executor.executeAsync(
              {input: inputTensor}, ['intermediate']);
          tfc.test_util.expectArraysClose(await result[0].data(), [3.0]);
        });

        it('should be able to execute control flow graph ' +
               'with intermediate node more than once',
           async () => {
             const inputTensor = tfc.scalar(1);

             const result = await executor.executeAsync(
                 {intermediate: inputTensor}, ['output:1']);
             tfc.test_util.expectArraysClose(await result[0].data(), [1]);
             const result2 = await executor.executeAsync(
                 {intermediate: inputTensor}, ['output:1']);
             tfc.test_util.expectArraysClose(await result2[0].data(), [1]);
           });

        it('should not have mem leak', async () => {
          const inputTensor = tfc.scalar(1);
          const numTensors: number = tfc.memory().numTensors;

          await executor.executeAsync({input: inputTensor}, ['output:1']);
          expect(tfc.memory().numTensors).toEqual(numTensors + 1);
        });
      });

      describe('controlFlowV2_if', () => {
        beforeEach(() => {
          inputNode = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'input',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const inputNode2: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'x',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const inputNode3: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'y',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          outputNode = {
            inputNames: ['input', 'x', 'y'],
            inputs: [inputNode, inputNode2, inputNode3],
            children: [],
            name: 'output',
            op: 'StatelessIf',
            category: 'control',
            attrParams: {
              'thenBranch': {'value': 'trueFunc', 'type': 'func'},
              'elseBranch': {'value': 'falseFunc', 'type': 'func'}
            },
            inputParams: {
              'cond': {'type': 'tensor', 'inputIndexStart': 0},
              'args': {
                'type': 'tensors',
                'inputIndexStart': 1,
                'inputIndexEnd': 0
              }
            }
          };
          inputNode.children.push(outputNode);
          inputNode2.children.push(outputNode);
          inputNode3.children.push(outputNode);
          const xNode: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'x',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const yNode: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'y',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const trueFuncGraph: Graph = {
            inputs: [xNode, yNode],
            nodes: {'x': xNode, 'y': yNode},
            outputs: [xNode],
            weights: [],
            placeholders: [xNode, yNode],
          };
          const falseFuncGraph: Graph = {
            inputs: [xNode, yNode],
            nodes: {'x': xNode, 'y': yNode},
            outputs: [yNode],
            weights: [],
            placeholders: [xNode, yNode],
          };
          graphWithControlFlow = {
            inputs: [inputNode, inputNode2, inputNode3],
            nodes: {
              'input': inputNode,
              'x': inputNode2,
              'y': inputNode3,
              'output': outputNode
            },
            outputs: [outputNode],
            weights: [],
            placeholders: [inputNode, inputNode2, inputNode3],
            functions: {trueFunc: trueFuncGraph, falseFunc: falseFuncGraph}
          };

          executor = new GraphExecutor(graphWithControlFlow);
          executor.weightMap = {};
        });

        it('should execute control flow v2 graph', async () => {
          const condTensor = tfc.scalar(true, 'bool');
          const condTensor2 = tfc.scalar(false, 'bool');
          const trueTensor = tfc.scalar(1, 'int32');
          const falseTensor = tfc.scalar(0, 'int32');

          let result = await executor.executeAsync(
              {input: condTensor, x: trueTensor, y: falseTensor}, ['output']);
          tfc.test_util.expectArraysClose(await result[0].data(), 1);
          result = await executor.executeAsync(
              {input: condTensor2, x: trueTensor, y: falseTensor}, ['output']);
          tfc.test_util.expectArraysClose(await result[0].data(), 0);
        });
        it('should not have mem leak', async () => {
          const condTensor = tfc.scalar(true, 'bool');
          const trueTensor = tfc.scalar(1, 'int32');
          const falseTensor = tfc.scalar(0, 'int32');
          const numTensors: number = tfc.memory().numTensors;

          await executor.executeAsync(
              {input: condTensor, x: trueTensor, y: falseTensor}, ['output']);
          expect(tfc.memory().numTensors).toEqual(numTensors);
        });
      });

      describe('controlFlowV2_while', () => {
        beforeEach(() => {
          const inputNode2: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'x',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const inputNode3: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'y',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          outputNode = {
            inputNames: ['x', 'y'],
            inputs: [inputNode2, inputNode3],
            children: [],
            name: 'output',
            op: 'StatelessWhile',
            category: 'control',
            attrParams: {
              'cond': {'value': 'condFunc', 'type': 'func'},
              'body': {'value': 'bodyFunc', 'type': 'func'}
            },
            inputParams: {
              'args': {
                'type': 'tensors',
                'inputIndexStart': 0,
                'inputIndexEnd': 0
              }
            }
          };
          inputNode2.children.push(outputNode);
          inputNode3.children.push(outputNode);
          const xNode: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'x',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const yNode: Node = {
            inputNames: [],
            inputs: [],
            children: [],
            name: 'y',
            op: 'Placeholder',
            category: 'graph',
            attrParams: {},
            inputParams: {}
          };
          const addNode: Node = {
            inputNames: ['x', 'y'],
            inputs: [xNode, yNode],
            children: [],
            name: 'add',
            op: 'Add',
            category: 'arithmetic',
            inputParams: {'a': createTensorAttr(0), 'b': createTensorAttr(1)},
            attrParams: {}
          };
          xNode.children.push(addNode);
          yNode.children.push(addNode);
          const bodyFunc: Graph = {
            inputs: [xNode, yNode],
            nodes: {'x': xNode, 'y': yNode, add: addNode},
            outputs: [addNode, yNode],
            weights: [],
            placeholders: [xNode, yNode],
          };
          const condFunc: Graph = {
            inputs: [xNode, yNode],
            nodes: {'x': xNode, 'y': yNode},
            outputs: [xNode],
            weights: [],
            placeholders: [xNode, yNode],
          };
          graphWithControlFlow = {
            inputs: [inputNode2, inputNode3],
            nodes: {'x': inputNode2, 'y': inputNode3, 'output': outputNode},
            outputs: [outputNode],
            weights: [],
            placeholders: [inputNode2, inputNode3],
            functions: {condFunc, bodyFunc}
          };

          executor = new GraphExecutor(graphWithControlFlow);
          executor.weightMap = {};
        });

        it('should execute control flow v2 graph', async () => {
          const trueTensor = tfc.scalar(-1, 'int32');
          const falseTensor = tfc.scalar(1, 'int32');

          const result = await executor.executeAsync(
              {x: trueTensor, y: falseTensor}, ['output']);
          tfc.test_util.expectArraysClose(await result[0].data(), 0);
        });
        it('should not have mem leak', async () => {
          const trueTensor = tfc.scalar(-1, 'int32');
          const falseTensor = tfc.scalar(1, 'int32');
          const numTensors: number = tfc.memory().numTensors;

          await executor.executeAsync(
              {x: trueTensor, y: falseTensor}, ['output']);
          expect(tfc.memory().numTensors).toEqual(numTensors + 1);
        });
        it('should not have mem leak when add index', async () => {
          const trueTensor = tfc.scalar(-1, 'int32');
          const falseTensor = tfc.scalar(1, 'int32');
          const numTensors: number = tfc.memory().numTensors;

          await executor.executeAsync(
              {x: trueTensor, y: falseTensor}, ['output:0']);
          expect(tfc.memory().numTensors).toEqual(numTensors + 1);
        });
      });
    });
  });
});
