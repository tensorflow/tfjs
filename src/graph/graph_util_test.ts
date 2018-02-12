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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {Tensor, Scalar} from '../tensor';
// tslint:disable-next-line:max-line-length
import {ConstantNode, Graph, Node, PlaceholderNode, ReLUNode, SquareNode, SymbolicTensor, VariableNode} from './graph';
import * as graph_util from './graph_util';
import {TensorArrayMap} from './tensor_array_map';

class TestNode extends Node {
  validate() {}
}

describe('graph_util.getUnorderedEvaluationSet', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('returns only node in graph', () => {
    const n = new TestNode(g, '', {}, new SymbolicTensor([]));
    const path = graph_util.getUnorderedEvaluationSet([n], []);
    expect(path.length).toEqual(1);
    expect(path[0]).toBe(n);
  });

  it('returns both nodes in graph with two connected nodes', () => {
    const t = new SymbolicTensor([]);
    const s = new TestNode(g, '', {}, t);
    const e = new TestNode(g, '', {'t': t}, new SymbolicTensor([]));
    const path = graph_util.getUnorderedEvaluationSet([e], []);
    expect(path.length).toEqual(2);
    expect(path).toContain(s);
    expect(path).toContain(e);
  });

  it('adds nodes in the termination set', () => {
    const t0 = new SymbolicTensor([]);
    const n0 = new TestNode(g, '', {}, t0);
    const t1 = new SymbolicTensor([]);
    const n1 = new TestNode(g, '', {'t0': t0}, t1);
    const n2 = new TestNode(g, '', {'t1': t1}, new SymbolicTensor([]));
    const path = graph_util.getUnorderedEvaluationSet([n2], [n0]);
    expect(path.length).toEqual(3);
    expect(path).toContain(n0);
    expect(path).toContain(n1);
    expect(path).toContain(n2);
  });

  it('does not process inputs from nodes in the termination set', () => {
    const t0 = new SymbolicTensor([]);
    const t1 = new SymbolicTensor([]);
    const n1 = new TestNode(g, '', {'t0': t0}, t1);
    const n2 = new TestNode(g, '', {'t1': t1}, new SymbolicTensor([]));
    const path = graph_util.getUnorderedEvaluationSet([n2], [n1]);
    expect(path.length).toEqual(2);
    expect(path).toContain(n1);
    expect(path).toContain(n2);
  });

  it('accumulates multiple inputs from nodes', () => {
    const t0 = new SymbolicTensor([]);
    const i0 = new TestNode(g, '', {}, t0);
    const t1 = new SymbolicTensor([]);
    const i1 = new TestNode(g, '', {}, t1);
    const n = new TestNode(g, '', {'t0': t0, 't1': t1}, new SymbolicTensor([]));
    const path = graph_util.getUnorderedEvaluationSet([n], []);
    expect(path.length).toEqual(3);
    expect(path).toContain(i0);
    expect(path).toContain(i1);
    expect(path).toContain(n);
  });

  it('enqueues each node once even if there are multiple paths to it', () => {
    const t0 = new SymbolicTensor([]);
    const n0 = new TestNode(g, '', {}, t0);
    const t1 = new SymbolicTensor([]);
    const n1 = new TestNode(g, '', {'t0': t0}, t1);
    const t2 = new SymbolicTensor([]);
    const n2 = new TestNode(g, '', {'t0': t0}, t2);
    const n3 =
        new TestNode(g, '', {'t1': t1, 't2': t2}, new SymbolicTensor([]));
    const set = graph_util.getUnorderedEvaluationSet([n3], []);
    expect(set.length).toEqual(4);
    expect(set).toContain(n0);
    expect(set).toContain(n1);
    expect(set).toContain(n2);
    expect(set).toContain(n3);
  });
});

describe('graph_util.getOrderedEvaluationSet', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('returns only node in unordered set', () => {
    const n = new TestNode(g, '', {}, new SymbolicTensor([]));
    expect(graph_util.getOrderedEvaluationSet([n])).toEqual([n]);
  });

  it('orders dependencies first (2 nodes)', () => {
    // 0 => 1:  [0 1]
    const t0 = new SymbolicTensor([]);
    const n0 = new TestNode(g, '', {}, t0);
    const n1 = new TestNode(g, '', {'t0': t0}, new SymbolicTensor([]));
    const unordered = [n1, n0];
    const ordered = [n0, n1];
    expect(graph_util.getOrderedEvaluationSet(unordered)).toEqual(ordered);
  });

  it('orders dependencies first (3 nodes)', () => {
    // 0 => 1, 1 => 2, 0 => 2:  [0 1 2]
    const t0 = new SymbolicTensor([]);
    const n0 = new TestNode(g, '', {}, t0);
    const t1 = new SymbolicTensor([]);
    const n1 = new TestNode(g, '', {'t0': t0}, t1);
    const n2 =
        new TestNode(g, '', {'t0': t0, 't1': t1}, new SymbolicTensor([]));
    const unordered = [n1, n2, n0];
    const ordered = [n0, n1, n2];
    expect(graph_util.getOrderedEvaluationSet(unordered)).toEqual(ordered);
  });

  it('orders dependencies first (5 nodes)', () => {
    // 0 => 1, 0 => 2, 0 => 4
    // 1 => 3
    // 2 => 3
    // 3 => 4
    // [0 1 2 3 4] or [0 2 1 3 4]
    const t0 = new SymbolicTensor([]);
    const n0 = new TestNode(g, '', {}, t0);
    const t1 = new SymbolicTensor([]);
    const n1 = new TestNode(g, '', {'t0': t0}, t1);
    const t2 = new SymbolicTensor([]);
    const n2 = new TestNode(g, '', {'t0': t0}, t2);
    const t3 = new SymbolicTensor([]);
    const n3 = new TestNode(g, '', {'t1': t1, 't2': t2}, t3);
    const n4 =
        new TestNode(g, '', {'t0': t0, 't3': t3}, new SymbolicTensor([]));
    const path = graph_util.getOrderedEvaluationSet([n4, n3, n2, n1, n0]);
    expect(path[0]).toBe(n0);
    const n2n1 = (path[1] === n2) && (path[2] === n1);
    const n1n2 = (path[1] === n1) && (path[2] === n2);
    expect(n2n1 || n1n2).toBe(true);
    expect(path[3]).toBe(n3);
    expect(path[4]).toBe(n4);
  });
});

describe('graph_util.isInputNode', () => {
  let g: Graph;
  let nda: Tensor;

  beforeEach(() => {
    g = new Graph();
    nda = dl.zeros([1]);
  });

  it('returns true for VariableNode', () => {
    expect(graph_util.isInputNode(new VariableNode(g, '', nda))).toEqual(true);
  });

  it('returns true for PlaceholderNode', () => {
    expect(graph_util.isInputNode(new PlaceholderNode(g, '', [1])))
        .toEqual(true);
  });

  it('returns true for ConstantNode', () => {
    expect(graph_util.isInputNode(new ConstantNode(g, dl.zeros([1]))))
        .toEqual(true);
  });

  it('returns false for ReLUNode', () => {
    expect(graph_util.isInputNode(new ReLUNode(g, new SymbolicTensor([]))))
        .toEqual(false);
  });
});

describe('graph_util.isPassthroughNode', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('returns false for a node that produces new NDArray', () => {
    const x = g.placeholder('x', []);
    const node = new SquareNode(g, x);
    const map = new TensorArrayMap();
    const xVal = Scalar.new(3);
    map.set(x, xVal);
    const yVal = Scalar.new(9);
    map.set(node.output, yVal);

    expect(graph_util.isPassthroughNode(node, map)).toBe(false);
    xVal.dispose();
    yVal.dispose();
  });
});
