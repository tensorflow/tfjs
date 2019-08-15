/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import {scalar} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../data/types';
import {Graph, Node} from '../operations/types';

import {rewritePrelu} from './model_rewrite';

describe('rewritePrelu', () => {
  it('should not rewrite non prelu add node', () => {
    const X_NODE: Node = {
      name: 'x',
      op: 'Const',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };

    const Y_NODE: Node = {
      name: 'y',
      op: 'Const',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };

    const ADD_NODE: Node = {
      name: 'add',
      op: 'Add',
      category: 'basic_math',
      inputNames: ['x', 'y'],
      inputs: [X_NODE, Y_NODE],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const graph: Graph = {
      placeholders: [],
      weights: [],
      inputs: [],
      outputs: [],
      nodes: {'add': ADD_NODE, 'x': X_NODE, 'y': Y_NODE}
    };

    const weights: NamedTensorsMap = {'x': [scalar(1)], 'y': [scalar(2)]};
    rewritePrelu(graph, weights);

    expect(Object.keys(graph.nodes)).toEqual(jasmine.arrayWithExactContents([
      'x', 'y', 'add'
    ]));
    expect(ADD_NODE.inputNames).toEqual(jasmine.arrayWithExactContents([
      'x', 'y'
    ]));
    expect(Object.keys(weights)).toEqual(jasmine.arrayWithExactContents([
      'x', 'y'
    ]));
  });
  it('should rewrite prelu add node', () => {
    const X_NODE: Node = {
      name: 'x',
      op: 'Placeholder',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };

    const Y_NODE: Node = {
      name: 'y',
      op: 'Const',
      category: 'graph',
      inputNames: [],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };

    const NEG_NODE: Node = {
      name: 'neg',
      op: 'Neg',
      category: 'basic_math',
      inputNames: ['x'],
      inputs: [X_NODE],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const RELU_NODE: Node = {
      name: 'relu',
      op: 'Relu',
      category: 'basic_math',
      inputNames: ['x'],
      inputs: [X_NODE],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const RELU_NODE2: Node = {
      name: 'relu2',
      op: 'Relu',
      category: 'basic_math',
      inputNames: ['neg'],
      inputs: [NEG_NODE],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const MUL_NODE: Node = {
      name: 'mul',
      op: 'Mul',
      category: 'basic_math',
      inputNames: ['y', 'relu2'],
      inputs: [Y_NODE, RELU_NODE2],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const NOOP_NODE: Node = {
      name: 'noop',
      op: 'Noop',
      category: 'basic_math',
      inputNames: ['add2'],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
    const PRELU_ADD_NODE: Node = {
      name: 'add2',
      op: 'Add',
      category: 'basic_math',
      inputNames: ['relu', 'mul'],
      inputs: [RELU_NODE, MUL_NODE],
      inputParams: {},
      attrParams: {},
      children: [NOOP_NODE]
    };
    const graph: Graph = {
      placeholders: [X_NODE],
      weights: [Y_NODE],
      inputs: [X_NODE],
      outputs: [NOOP_NODE],
      nodes: {
        'add2': PRELU_ADD_NODE,
        'x': X_NODE,
        'y': Y_NODE,
        'neg': NEG_NODE,
        'relu': RELU_NODE,
        'relu2': RELU_NODE2,
        'mul': MUL_NODE,
        'noop': NOOP_NODE
      }
    };
    X_NODE.children = [RELU_NODE, NEG_NODE];
    Y_NODE.children = [MUL_NODE];

    const weights: NamedTensorsMap = {'x': [scalar(1)], 'y': [scalar(2)]};
    rewritePrelu(graph, weights);
    expect(Object.keys(graph.nodes)).toEqual(jasmine.arrayWithExactContents([
      'x', 'y', 'y_neg', 'add2_Prelu', 'noop'
    ]));

    expect(graph.nodes['y'].children).toEqual([]);
    expect(graph.nodes['y_neg'].inputNames).toEqual([]);
    expect(graph.nodes['y_neg'].op).toEqual('Const');
    expect(graph.nodes['x'].children).toEqual([graph.nodes['add2_Prelu']]);
    expect(graph.nodes['add2_Prelu'].inputNames).toEqual(['x', 'y_neg']);
    expect(graph.nodes['add2_Prelu'].children).toEqual([graph.nodes['noop']]);
    expect(graph.nodes['add2_Prelu'].op).toEqual('Prelu');
    expect(graph.nodes['noop'].inputNames).toEqual(['add2_Prelu']);
    expect(Object.keys(weights)).toEqual(['x', 'y', 'y_neg']);
  });
});
