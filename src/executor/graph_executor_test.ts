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

import * as operations from '../operations/index';

import {ExecutionContext, GraphExecutor} from './index';

let executor: GraphExecutor;
let inputNode: operations.Node;
let constNode: operations.Node;
let outputNode: operations.Node;
let graph: operations.Graph;
let graphWithControlFlow: operations.Graph;

describe('GraphExecutor', () => {
  beforeEach(() => {
    inputNode = {
      inputNames: [],
      inputs: [],
      children: [],
      name: 'input',
      op: 'placeholder',
      category: 'graph',
      params: {}
    };
    constNode = {
      inputNames: [],
      inputs: [],
      children: [],
      name: 'const',
      op: 'const',
      category: 'graph',
      params: {}
    };
    outputNode = {
      inputNames: ['input', 'const'],
      inputs: [inputNode, constNode],
      children: [],
      name: 'output',
      op: 'add',
      category: 'arithmetic',
      params: {}
    };
    graph = {
      inputs: [constNode, inputNode],
      nodes: {'input': inputNode, 'const': constNode, 'output': outputNode},
      outputs: [outputNode],
      withControlFlow: false,
      placeholders: [inputNode]
    };
    inputNode.children.push(outputNode);
    constNode.children.push(outputNode);
    executor = new GraphExecutor(graph);
  });
  afterEach(() => {});

  describe('execute graph', () => {
    describe('initialization', () => {
      it('should expose placehoder', () => {
        expect(executor.inputNodes).toEqual(['input']);
      });

      it('should expose output', () => {
        expect(executor.outputNodes).toEqual(['output']);
      });
    });

    describe('graph level', () => {
      it('should execute the op', () => {
        executor = new GraphExecutor(graph);
        const inputTensor = tfc.scalar(1);
        const constTensor = tfc.scalar(2);
        const spy =
            spyOn(operations, 'executeOp')
                .and.callFake((node: operations.Node) => {
                  return node.op === 'const' ? [constTensor] : [inputTensor];
                });

        executor.execute({input: [inputTensor]});

        expect(spy.calls.allArgs()).toEqual([
          [inputNode, jasmine.any(Object), jasmine.any(ExecutionContext)],
          [constNode, jasmine.any(Object), jasmine.any(ExecutionContext)],
          [outputNode, jasmine.any(Object), jasmine.any(ExecutionContext)]
        ]);
      });

      it('should execute control flow graph', async (done) => {
        inputNode = {
          inputNames: [],
          inputs: [],
          children: [],
          name: 'input',
          op: 'placeholder',
          category: 'graph',
          params: {}
        };
        constNode = {
          inputNames: [],
          inputs: [],
          children: [],
          name: 'const',
          op: 'const',
          category: 'graph',
          params: {}
        };
        outputNode = {
          inputNames: ['input', 'const'],
          inputs: [inputNode, constNode],
          children: [],
          name: 'output',
          op: 'switch',
          category: 'control',
          params: {}
        };
        inputNode.children.push(outputNode);
        constNode.children.push(outputNode);
        graphWithControlFlow = {
          inputs: [constNode, inputNode],
          nodes: {'input': inputNode, 'const': constNode, 'output': outputNode},
          outputs: [outputNode],
          withControlFlow: true,
          placeholders: [inputNode]
        };

        executor = new GraphExecutor(graphWithControlFlow);
        const inputTensor = tfc.scalar(1);
        const constTensor = tfc.scalar(2);
        executor.weightMap = {const : [constTensor]};
        const spy =
            spyOn(operations, 'executeOp')
                .and.callFake((node: operations.Node) => {
                  return node.op === 'const' ? [constTensor] : [inputTensor];
                });

        await executor.executeAsync({input: [inputTensor]}).then(result => {
          expect(spy.calls.allArgs()).toEqual([
            [inputNode, jasmine.any(Object), jasmine.any(ExecutionContext)],
            [outputNode, jasmine.any(Object), jasmine.any(ExecutionContext)],
            [constNode, jasmine.any(Object), jasmine.any(ExecutionContext)],
          ]);
          done();
        });

        it('should throw exception if missing inputs', () => {
          expect(() => executor.execute({}))
              .toThrow(new Error('Missing input placeholders: input'));
        });

        it('should throw exception if contains extra inputs', () => {
          const inputTensor = tfc.scalar(1);
          expect(() => executor.execute({
            test: [inputTensor],
            input: [inputTensor]
          })).toThrow(new Error('Extra input tensors: test'));
        });
      });
    });
  });
});
