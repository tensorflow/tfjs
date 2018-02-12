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

// tslint:disable-next-line:max-line-length
import {InputProvider} from '../data/input_provider';
import {ENV} from '../environment';
import * as dl from '../index';
import {Tensor} from '../tensor';
// tslint:disable-next-line:max-line-length
import {ConstantNode, Graph, Node, PlaceholderNode, SymbolicTensor, VariableNode} from './graph';
import {FeedDictionary, FeedEntry} from './session';
import * as session_util from './session_util';
import {TensorArrayMap} from './tensor_array_map';

class TestNode extends Node {
  validate() {}
}

describe('getTerminatingNodesFromFeedDictionary', () => {
  it('returns an empty node array from an empty FeedDictionary', () => {
    expect(session_util.getTerminatingNodesFromFeedDictionary(
               new FeedDictionary()))
        .toEqual([]);
  });

  it('returns the only node in the feed dictionary', () => {
    dl.tidy(() => {
      const node = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const fd =
          new FeedDictionary([{tensor: node.output, data: dl.zeros([1])}]);
      expect(session_util.getTerminatingNodesFromFeedDictionary(fd)).toEqual([
        node
      ]);
    });
  });

  it('returns every node from the feed dictionary', () => {
    dl.tidy(() => {
      const n0 = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const n1 = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const n2 = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const n3 = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const n4 = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
      const feeds: FeedEntry[] = [
        {tensor: n0.output, data: dl.zeros([1])},
        {tensor: n1.output, data: dl.zeros([1])},
        {tensor: n2.output, data: dl.zeros([1])},
        {tensor: n3.output, data: dl.zeros([1])},
        {tensor: n4.output, data: dl.zeros([1])}
      ];
      const fd = new FeedDictionary(feeds);
      const nodes = session_util.getTerminatingNodesFromFeedDictionary(fd);
      expect(nodes).toContain(n0);
      expect(nodes).toContain(n1);
      expect(nodes).toContain(n2);
      expect(nodes).toContain(n3);
      expect(nodes).toContain(n4);
    });
  });
});

describe('addPersistentArraysToTensorArrayMap', () => {
  let map: TensorArrayMap;
  let g: Graph;

  beforeEach(() => {
    map = new TensorArrayMap();
    g = new Graph();
  });

  it('does nothing with empty evaluationSet', () => {
    session_util.addPersistentArraysToTensorArrayMap([], map);
    expect(map.size()).toEqual(0);
  });

  it('adds the only VariableNode to the map', () => {
    const v = new VariableNode(g, '', dl.zeros([1]));
    session_util.addPersistentArraysToTensorArrayMap([v], map);
    expect(map.get(v.output)).toBe(v.data);
  });

  it('adds the only ConstantNode to the map', () => {
    const c = new ConstantNode(g, dl.zeros([1]));
    session_util.addPersistentArraysToTensorArrayMap([c], map);
    expect(map.get(c.output)).toBe(c.data);
  });

  it('does nothing with nodes that aren\'t VariableNodes or ConstantNodes',
     () => {
       const nodes = [new TestNode(g, '', {}, new SymbolicTensor([]))];
       session_util.addPersistentArraysToTensorArrayMap(nodes, map);
       expect(map.size()).toEqual(0);
     });

  it('adds multiple VariableNodes to the map', () => {
    const nodes = [
      new VariableNode(g, '', dl.zeros([1])),
      new VariableNode(g, '', dl.zeros([1])),
      new VariableNode(g, '', dl.zeros([1]))
    ];
    session_util.addPersistentArraysToTensorArrayMap(nodes, map);
    expect(map.get(nodes[0].output)).toBe(nodes[0].data);
    expect(map.get(nodes[1].output)).toBe(nodes[1].data);
    expect(map.get(nodes[2].output)).toBe(nodes[2].data);
  });

  it('adds multiple ConstantNodes to the map', () => {
    dl.tidy(() => {
      const nodes = [
        new ConstantNode(g, dl.zeros([1])), new ConstantNode(g, dl.zeros([1])),
        new ConstantNode(g, dl.zeros([1]))
      ];
      session_util.addPersistentArraysToTensorArrayMap(nodes, map);
      expect(map.get(nodes[0].output)).toBe(nodes[0].data);
      expect(map.get(nodes[1].output)).toBe(nodes[1].data);
      expect(map.get(nodes[2].output)).toBe(nodes[2].data);
    });
  });

  it('skips non-VariableNode or ConstantNode entries in the set', () => {
    const nodes: Node[] = [
      new TestNode(g, '', {}, new SymbolicTensor([])),
      new VariableNode(g, '', dl.zeros([1])),
      new TestNode(g, '', {}, new SymbolicTensor([])),
      new ConstantNode(g, dl.zeros([1])),
      new TestNode(g, '', {}, new SymbolicTensor([])),
      new VariableNode(g, '', dl.zeros([1]))
    ];
    session_util.addPersistentArraysToTensorArrayMap(nodes, map);
    expect(map.size()).toEqual(3);
    expect(map.get(nodes[1].output)).toBe((nodes[1] as VariableNode).data);
    expect(map.get(nodes[3].output)).toBe((nodes[3] as ConstantNode).data);
    expect(map.get(nodes[5].output)).toBe((nodes[5] as VariableNode).data);
  });
});

describe('loadInputsFromFeedDictionaryToTensorArrayMap', () => {
  let map: TensorArrayMap;
  const math = ENV.math;

  beforeEach(() => {
    map = new TensorArrayMap();
  });

  it('does nothing with empty feed dictionary', () => {
    const fd = new FeedDictionary();
    session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(0);
  });

  it('adds the only NDArray feed dict entry to the map', () => {
    const tensor = new SymbolicTensor([1]);
    const fd = new FeedDictionary([{tensor, data: dl.zeros([1])}]);
    session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(1);
    expect(map.get(tensor)).toBe(fd.dict[tensor.id].data as Tensor);
  });

  it('adds the only provider feed dict entry to the map', () => {
    const tensor = new SymbolicTensor([2]);
    const ndarray = dl.zeros([2]);
    const provider: InputProvider = {
      getNextCopy():
          Tensor {  // Don't return a copy in this case so we can test
            // that we returned the
            // right value.
            return ndarray;
          },
      // No need to dispose when not using the webgl backend.
      disposeCopy() {}
    };
    const fd = new FeedDictionary([{tensor, data: provider}]);
    session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(1);
    expect(map.get(tensor)).toBe(ndarray);
  });

  it('adds every NDArray feed dict entry to the map', () => {
    const tensors = [
      new SymbolicTensor([1]), new SymbolicTensor([1]), new SymbolicTensor([1]),
      new SymbolicTensor([1]), new SymbolicTensor([1])
    ];
    const feeds = tensors.map(tensor => {
      return {tensor, data: dl.zeros([1])};
    });
    const fd = new FeedDictionary(feeds);
    session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(tensors.length);
    tensors.forEach(
        tensor =>
            expect(map.get(tensor)).toBe(fd.dict[tensor.id].data as Tensor));
  });

  it('adds every provider feed dict entry to the map', () => {
    const tensors = [
      new SymbolicTensor([1]), new SymbolicTensor([1]), new SymbolicTensor([1]),
      new SymbolicTensor([1]), new SymbolicTensor([1])
    ];
    const ndarrays: Tensor[] = [];
    for (let i = 0; i < tensors.length; i++) {
      ndarrays.push(dl.zeros([1]));
    }
    let idx = 0;
    const provider: InputProvider = {
      getNextCopy(): Tensor {
        const ndarray = ndarrays[idx];
        idx++;
        return ndarray;
      },
      disposeCopy() {}
    };

    const feeds: FeedEntry[] = [];
    for (let i = 0; i < tensors.length; i++) {
      feeds.push({tensor: tensors[i], data: provider});
    }

    const fd = new FeedDictionary(feeds);
    session_util.loadInputsFromFeedDictionaryToTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(tensors.length);
    for (let i = 0; i < tensors.length; i++) {
      expect(map.get(tensors[i])).toBe(ndarrays[i]);
    }
  });

  it('throws when provides data that does not match tensor shape', () => {
    const tensor = new SymbolicTensor([4, 5]);
    const fd = new FeedDictionary([{tensor, data: dl.zeros([2, 3])}]);
    expect(
        () => session_util.loadInputsFromFeedDictionaryToTensorArrayMap(
            fd, map, math))
        .toThrowError();
  });
});

describe('releaseFeedDictionaryInputsFromTensorArrayMap', () => {
  let map: TensorArrayMap;
  const math = ENV.math;

  beforeEach(() => {
    map = new TensorArrayMap();
  });

  it('doesn\'t remove anything when feed dictionary is empty', () => {
    map.set(new SymbolicTensor([]), null);
    const fd = new FeedDictionary();
    session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(1);
  });

  it('doesn\'t remove tensors from map that don\'t exist in feed', () => {
    const fdTensor = new SymbolicTensor([]);
    const nda = dl.zeros([1]);
    const fd = new FeedDictionary([{tensor: fdTensor, data: dl.zeros([1])}]);
    const nonFDTensor = new SymbolicTensor([]);
    map.set(nonFDTensor, nda);
    session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(1);
    expect(map.get(nonFDTensor)).toBe(nda);
  });

  it('removes only tensor in map and feed dict', () => {
    const tensor = new SymbolicTensor([]);
    const ndarray = dl.zeros([1]);
    const fd = new FeedDictionary([{tensor, data: ndarray}]);
    map.set(tensor, ndarray);
    session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(0);
  });

  it('removes from map all tensors in feed dict', () => {
    const tensors = [
      new SymbolicTensor([]), new SymbolicTensor([]), new SymbolicTensor([])
    ];

    const feeds = tensors.map(tensor => {
      return {tensor, data: dl.zeros([1])};
    });
    const fd = new FeedDictionary(feeds);
    tensors.forEach(
        tensor => map.set(tensor, fd.dict[tensor.id].data as Tensor));
    session_util.releaseFeedDictionaryInputsFromTensorArrayMap(fd, map, math);
    expect(map.size()).toEqual(0);
  });
});

describe('disposeAndInitializeOperationOutputs', () => {
  let map: TensorArrayMap;
  let g: Graph;
  beforeEach(() => {
    map = new TensorArrayMap();
    g = new Graph();
  });

  it('does nothing to map if set is empty', () => {
    session_util.disposeAndInitializeOperationOutputs([], map);
    expect(map.size()).toEqual(0);
  });

  it('does nothing to map if set has no input nodes', () => {
    const nodes = [
      new VariableNode(g, '', dl.zeros([1])), new PlaceholderNode(g, '', [1])
    ];
    session_util.disposeAndInitializeOperationOutputs(nodes, map);
    expect(map.size()).toEqual(0);
  });

  it('adds output tensor from only operation node', () => {
    const input = new SymbolicTensor([]);
    const t = new SymbolicTensor([]);
    session_util.disposeAndInitializeOperationOutputs(
        [new TestNode(g, '', {'in': input}, t)], map);
    expect(map.size()).toEqual(1);
    expect(map.hasNullArray(t)).toEqual(true);
  });

  it('adds output tensors from all operation nodes', () => {
    const input = new SymbolicTensor([]);
    const tensors = [
      new SymbolicTensor([]), new SymbolicTensor([]), new SymbolicTensor([])
    ];
    const nodes: Node[] = [];
    tensors.forEach(
        tensor => nodes.push(new TestNode(g, '', {'in': input}, tensor)));
    session_util.disposeAndInitializeOperationOutputs(nodes, map);
    expect(map.size()).toEqual(nodes.length);
    tensors.forEach(tensor => expect(map.hasNullArray(tensor)).toEqual(true));
  });
});

describe('removeFeedDictionaryNodesFromEvaluationSet', () => {
  let set: Node[];

  beforeEach(() => {
    set = [];
  });

  it('does nothing when feed dictionary is empty', () => {
    const node = new TestNode(new Graph(), '', {}, new SymbolicTensor([]));
    set.push(node);
    const fd = new FeedDictionary();
    session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
    expect(set.length).toEqual(1);
    expect(set[0]).toBe(node);
  });

  it('removes only feed dict node from set', () => {
    set.push(new TestNode(new Graph(), '', {}, new SymbolicTensor([])));
    const fd =
        new FeedDictionary([{tensor: set[0].output, data: dl.zeros([1])}]);
    session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
    expect(set.length).toEqual(0);
  });

  it('removes only feed dict nodes from set', () => {
    const g = new Graph();
    const remainingNodes = [
      new TestNode(g, '', {}, new SymbolicTensor([])),
      new TestNode(g, '', {}, new SymbolicTensor([])),
      new TestNode(g, '', {}, new SymbolicTensor([]))
    ];

    set.push(remainingNodes[0]);
    set.push(new TestNode(g, '', {}, new SymbolicTensor([])));
    const feeds: FeedEntry[] = [];
    feeds.push({tensor: set[set.length - 1].output, data: dl.zeros([1])});
    set.push(remainingNodes[1]);
    set.push(new TestNode(g, '', {}, new SymbolicTensor([])));
    feeds.push({tensor: set[set.length - 1].output, data: dl.zeros([1])});
    set.push(remainingNodes[2]);

    const fd = new FeedDictionary(feeds);
    session_util.removeFeedDictionaryNodesFromEvaluationSet(fd, set);
    expect(set).toEqual(remainingNodes);
  });
});

describe('throwErrorIfEvaluationSetContainsPlaceholderNodes', () => {
  let g: Graph;
  beforeEach(() => g = new Graph());

  it('doesn\'t throw on an empty node array', () => {
    session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([]);
  });

  it('doesn\'t throw if array contains non-placeholder nodes', () => {
    session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(
        [new TestNode(g, '', {}, new SymbolicTensor([]))]);
  });

  it('throws if the array only contains a placeholder node', () => {
    expect(
        () => session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(
            [new PlaceholderNode(g, '', [])]))
        .toThrowError(/Placeholder node/);
  });

  it('thrown error contains the tensor shape', () => {
    expect(
        () => session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(
            [new PlaceholderNode(g, '', [1, 2, 3, 4, 5])]))
        .toThrowError(/[1, 2, 3, 4, 5]/);
  });

  it('throws if the non-first element in the array is a placeholder', () => {
    expect(
        () => session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes([
          new TestNode(g, '', {}, new SymbolicTensor([])),
          new PlaceholderNode(g, '', [])
        ]))
        .toThrowError(/Placeholder node/);
  });
});
