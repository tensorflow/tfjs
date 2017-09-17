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

import {ConstantNode, Node, Tensor} from './graph';
import * as priority_queue from './priority_queue';
import {PriorityQueue} from './priority_queue';
import {TensorArrayMap} from './tensor_array_map';

/**
 * Given a target node in a graph, accumulate the set of all nodes that need to
 * be evaluated in order to evaluate the target graph. Traversal stops anywhere
 * a node's values are fed in externally via "feed dicts".
 * @param nodes The nodes to be evaluated.
 * @param terminatingNodes The set of nodes that stop traversal.
 * @return The unordered set of nodes that need to be evaluated.
 */
export function getUnorderedEvaluationSet(
    nodes: Node[], terminatingNodes: Node[]): Node[] {
  const terminatingNodeMap: {[id: number]: Node} = {};
  const seen: {[id: number]: Node} = {};
  const set: Node[] = [];
  const visit: Node[] = nodes.slice();
  terminatingNodes.forEach(node => terminatingNodeMap[node.id] = node);
  /* Flood fill: While the 'to visit' stack is not empty, pop a node off of it.
   * If the node has not yet been visited, add it to the set, mark it as seen,
   * and enqueue all of its ancestor (input) nodes. */
  while (visit.length !== 0) {
    const cur = visit.pop();
    if (seen[cur.id] == null) {
      if (terminatingNodeMap[cur.id] == null) {
        Object.keys(cur.inputs)
            .map(inputName => cur.inputs[inputName])
            .forEach(input => visit.push(input.node));
      }
      set.push(cur);
      seen[cur.id] = cur;
    }
  }
  return set;
}

/**
 * Given a set of nodes, compute their order such that all dependent nodes are
 * evaluated after their dependees. This is the 'inference order' for nodes in
 * the operation graph.
 * @param unorderedEvaluationSet The unordered set of nodes that need to be
 * evaluated.
 * @return The input nodes in forward evaluation order.
 */
export function getOrderedEvaluationSet(unorderedEvaluationSet: Node[]):
    Node[] {
  /* A priority queue is used, where the priority is the remaining number of
   * unevaluated nodes whose inputs come from the element node. This guarantees
   * that all downstream nodes will be dequeued before their ancestors. */
  const set: Node[] = [];
  const nodeIndices: {[id: number]: number} = {};
  const pendingDependencies: {[id: number]: number} = {};

  /* The queue priority callback looks at the number of pending dependencies of
   * a given node. The queue index observer callback maintains the location of
   * each node in the array, for priority updates. */
  const nodeQueue = new PriorityQueue<Node>(
      (a: Node, b: Node) => priority_queue.defaultCompare(
          pendingDependencies[a.id], pendingDependencies[b.id]),
      (node: Node, newIndex: number) => nodeIndices[node.id] = newIndex);

  unorderedEvaluationSet.forEach(node => pendingDependencies[node.id] = 0);

  /* For every descendent of a node (output of ancestor is input to descendant),
   * increment the 'pending dependency count' for the ancestor. This prepares
   * the 'pending dependency count' as a priority map. */
  unorderedEvaluationSet.forEach(
      node => Object.keys(node.inputs)
                  .map(key => node.inputs[key])
                  .forEach(input => {
                    if (unorderedEvaluationSet.indexOf(input.node) !== -1) {
                      pendingDependencies[input.node.id]++;
                    }
                  }));

  unorderedEvaluationSet.forEach(node => nodeQueue.enqueue(node));

  while (!nodeQueue.empty()) {
    set.unshift(nodeQueue.dequeue());
    /* As each node is visited, decrement the 'pending dependency count' of
     * each ancestor, and tell the priority queue that the priority has changed.
     */
    Object.keys(set[0].inputs).map(key => set[0].inputs[key]).forEach(input => {
      if (unorderedEvaluationSet.indexOf(input.node) === -1) {
        return;
      }
      pendingDependencies[input.node.id]--;
      nodeQueue.update(input.node, nodeIndices[input.node.id]);
    });
  }

  return set;
}

/**
 * @return True iff the node is an input node.
 */
export function isInputNode(node: Node): boolean {
  return Object.keys(node.inputs).length === 0;
}

export function shouldBackProp(t: Tensor): boolean {
  return !(t.node instanceof ConstantNode);
}

export function isPassthroughNode(node: Node, map: TensorArrayMap): boolean {
  const keys = Object.keys(node.inputs);
  for (let i = 0; i < keys.length; i++) {
    const input = node.inputs[keys[i]];
    if (map.get(input, true) === map.get(node.output, true)) {
      return true;
    }
  }
  return false;
}
