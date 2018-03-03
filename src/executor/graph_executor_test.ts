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

import * as dl from 'deeplearn';
import * as operations from '../operations/index';
import {GraphExecutor} from './index';

let executor: GraphExecutor;
let inputNode: operations.Node;
let constNode: operations.Node;
let outputNode: operations.Node;
let graph: operations.Graph;

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
      outputs: [outputNode]
    };
    inputNode.children.push(outputNode);
    constNode.children.push(outputNode);
    executor = new GraphExecutor(graph);
  });
  afterEach(() => {});

  describe('execute graph', () => {
    describe('graph level', () => {
      it('should execute the op', () => {
        const inputTensor = dl.Scalar.new(1, 'float32');
        const constTensor = dl.Scalar.new(2, 'float32');
        const spy =
            spyOn(operations, 'executeOp')
                .and.callFake((node: operations.Node) => {
                  return node.op === 'const' ? constTensor : inputTensor;
                });

        executor.execute({input: inputTensor});

        expect(spy.calls.allArgs()).toEqual([
          [inputNode, jasmine.any(Object)], [constNode, jasmine.any(Object)],
          [outputNode, jasmine.any(Object)]
        ]);
      });
    });
  });
});
