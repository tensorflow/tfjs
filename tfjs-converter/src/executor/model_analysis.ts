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

import {NamedTensorMap} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../data/types';
import {parseNodeName} from '../operations/executors/utils';
import {Graph, Node} from '../operations/types';

export interface ExecutionInfo {
  inputs: NamedTensorMap;
  outputs: Node[];
  usedNodes: Set<string>;
  missingInputs: string[];
  dynamicNode: Node;
  syncInputs: string[];
}

/**
 * Given graph inputs and desired outputs, find the minimal set of nodes
 * to execute in order to compute the outputs. In addition return other useful
 * info such:
 * - Missing inputs needed to compute the output.
 * - Whether the subgraph contains dynamic ops (control flow, dynamic shape).
 * - Alternative inputs in order to avoid async (dynamic op) execution.
 */
export function getExecutionSubgraph(
    inputs: NamedTensorMap, outputs: Node[], weightMap: NamedTensorsMap,
    initNodes?: Node[]): ExecutionInfo {
  const usedNodes = new Set<string>();
  const missingInputs: string[] = [];
  let dynamicNode: Node = null;
  let syncInputs: string[] = null;

  // Start with the outputs, going backwards and find all the nodes that are
  // needed to compute those outputs.
  const seen = new Set<string>();
  const inputNodeNames =
      new Set(Object.keys(inputs).map((name) => parseNodeName(name)[0]));

  initNodes = initNodes || [];
  const initNodeNames =
      new Set(initNodes.map((node) => parseNodeName(node.name)[0]));

  const frontier = [...outputs];
  while (frontier.length > 0) {
    const node = frontier.pop();
    if (isControlFlow(node) || isDynamicShape(node) || isHashTable(node)) {
      if (dynamicNode == null) {
        dynamicNode = node;
        syncInputs = dynamicNode.children.map(child => child.name)
                         .filter(name => usedNodes.has(name));
      }
    }
    usedNodes.add(node.name);

    // Weights are dead end since we already have their values.
    if (weightMap[node.name] != null) {
      continue;
    }
    // This node is a dead end since it's one of the user-provided inputs.
    if (inputNodeNames.has(node.name)) {
      continue;
    }
    // This node is a dead end since it doesn't have any inputs.
    if (initNodeNames.has(node.name)) {
      continue;
    }
    if (node.inputs.length === 0) {
      missingInputs.push(node.name);
      continue;
    }
    node.inputs.forEach(input => {
      // Don't add to the frontier if it is already there.
      if (seen.has(input.name)) {
        return;
      }
      seen.add(input.name);
      frontier.push(input);
    });
  }
  return {inputs, outputs, usedNodes, missingInputs, dynamicNode, syncInputs};
}

/**
 * Given the execution info, return a list of nodes in topological order that
 * need to be executed to compute the output.
 */
export function getNodesInTopologicalOrder(
    graph: Graph, executionInfo: ExecutionInfo): Node[] {
  const {usedNodes, inputs} = executionInfo;
  const inputNodes = Object.keys(inputs)
                         .map(name => parseNodeName(name)[0])
                         .map(name => graph.nodes[name]);
  const initNodes = graph.initNodes || [];

  const isUsed = (node: Node|string) =>
      usedNodes.has(typeof node === 'string' ? node : node.name);

  function unique(nodes: Node[]): Node[] {
    return [...new Map(nodes.map((node) => [node.name, node])).values()];
  }
  const predefinedNodes = unique([
                            ...inputNodes,
                            ...graph.weights,
                            ...initNodes,
                          ]).filter(isUsed);
  const allNodes = unique([
                     ...predefinedNodes,
                     ...Object.values(graph.nodes),
                   ]).filter(isUsed);
  const nameToNode =
      new Map<string, Node>(allNodes.map((node) => [node.name, node]));

  const inCounts: Record<string, number> = {};
  for (const node of allNodes) {
    inCounts[node.name] = inCounts[node.name] || 0;
    for (const child of node.children) {
      // When the child is unused, set in counts to infinity so that it will
      // never be decreased to 0 and added to the execution list.
      if (!isUsed(child)) {
        inCounts[child.name] = Number.POSITIVE_INFINITY;
      }
      inCounts[child.name] = (inCounts[child.name] || 0) + 1;
    }
  }

  // Build execution order for all used nodes regardless whether they are
  // predefined or not.
  const frontier = Object.entries(inCounts)
                       .filter(([, inCount]) => inCount === 0)
                       .map(([name]) => name);
  const orderedNodeNames = [...frontier];
  while (frontier.length > 0) {
    const nodeName = frontier.pop();
    const node = nameToNode.get(nodeName)!;
    for (const child of node.children.filter(isUsed)) {
      if (--inCounts[child.name] === 0) {
        orderedNodeNames.push(child.name);
        frontier.push(child.name);
      }
    }
  }

  const orderedNodes = orderedNodeNames.map((name) => nameToNode.get(name));
  const filteredOrderedNodes =
      filterPredefinedReachableNodes(orderedNodes, predefinedNodes);

  // TODO: Turn validation on/off with tf env flag.
  validateNodesExecutionOrder(filteredOrderedNodes, predefinedNodes);

  return filteredOrderedNodes;
}

/**
 * This is a helper function of `getNodesInTopologicalOrder`.
 * Returns ordered nodes reachable by at least one predefined node.
 * This can help us filter out redundant nodes from the returned node list.
 * For example:
 * If we have four nodes with dependencies like this:
 *   a --> b --> c --> d
 * when node `c` is predefined (e.g. given as an input tensor), we can
 * skip node `a` and `b` since their outputs will never be used.
 *
 * @param orderedNodes Graph nodes in execution order.
 * @param predefinedNodes Graph inputs, weights, and init nodes. Nodes in this
 *     list must have distinct names.
 */
function filterPredefinedReachableNodes(
    orderedNodes: Node[], predefinedNodes: Node[]) {
  const nameToNode =
      new Map<string, Node>(orderedNodes.map((node) => [node.name, node]));

  // TODO: Filter out more nodes when >=2 nodes are predefined in a path.
  const stack = predefinedNodes.map((node) => node.name);
  const predefinedReachableNodeNames = new Set(stack);
  // Perform a DFS starting from the set of all predefined nodes
  // to find the set of all nodes reachable from the predefined nodes.
  while (stack.length > 0) {
    const nodeName = stack.pop();
    const node = nameToNode.get(nodeName)!;
    for (const child of node.children) {
      if (!nameToNode.has(child.name) ||
          predefinedReachableNodeNames.has(child.name)) {
        continue;
      }
      predefinedReachableNodeNames.add(child.name);
      stack.push(child.name);
    }
  }

  // Filter out unreachable nodes and build the ordered node list.
  const filteredOrderedNodes = orderedNodes.filter(
      (node) => predefinedReachableNodeNames.has(node.name));

  return filteredOrderedNodes;
}

class NodesExecutionOrderError extends Error {
  constructor(message: string) {
    super(`NodesExecutionOrderError: ${message}`);
  }
}

/**
 * This is a helper function of `getNodesInTopologicalOrder`.
 * Validates property: given nodes `a` and `b`, Order(a) > Order(b) if `a`
 * is a child of `b`. This function throws an error if validation fails.
 *
 * @param orderedNodes Graph nodes in execution order.
 * @param predefinedNodes Graph inputs, weights, and init nodes. Nodes in this
 *     list must have distinct names.
 */
function validateNodesExecutionOrder(
    orderedNodes: Node[], predefinedNodes: Node[]) {
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));
  const predefinedNodeNames = new Set(predefinedNodes.map((node) => node.name));
  const isPredefined = (node: Node|string) =>
      predefinedNodeNames.has(typeof node === 'string' ? node : node.name);
  const willBeExecutedNodeNames =
      new Set(orderedNodes.map((node) => node.name));
  const willBeExecuted = (node: Node|string) =>
      willBeExecutedNodeNames.has(typeof node === 'string' ? node : node.name);

  for (const node of orderedNodes) {
    for (const child of node.children.filter(willBeExecuted)) {
      if (!nodeNameToOrder.has(child.name)) {
        throw new NodesExecutionOrderError(
            `Child ${child.name} of node ${node.name} is unreachable.`);
      }
      if (nodeNameToOrder.get(node.name) > nodeNameToOrder.get(child.name)) {
        throw new NodesExecutionOrderError(`Node ${
            node.name} is scheduled to run after its child ${child.name}.`);
      }
    }
    if (!isPredefined(node)) {
      for (const input of node.inputs) {
        if (!nodeNameToOrder.has(input.name)) {
          throw new NodesExecutionOrderError(
              `Input ${input.name} of node ${node.name} is unreachable.`);
        }
        if (nodeNameToOrder.get(input.name) > nodeNameToOrder.get(node.name)) {
          throw new NodesExecutionOrderError(`Node ${
              node.name} is scheduled to run before its input ${input.name}.`);
        }
      }
    }
  }
}

/**
 * Given the execution info, return a map from node name to the disposable
 * node name list after its execution.
 *
 * @returns A map from node name to disposable nodes after its
 *     execution. That is, for a node `x`, `nodeLiveUntilMap[x]` indicates
 *     all nodes which their intermediate tensors should be disposed after `x`
 *     being executed.
 */
export function getNodeLiveUntilMap(orderedNodes: Node[]): Map<string, Node[]> {
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));

  const INF_LIFE = Number.MAX_SAFE_INTEGER;
  // Make control flow nodes (and consequently their direct parents)
  // live forever since they're tricky to track correctly.
  const selfLifespans = orderedNodes.map(
      (node, nodeOrder) => isControlFlow(node) ? INF_LIFE : nodeOrder);
  const getSelfLifeSpan = (node: Node) => {
    const selfLife = selfLifespans[nodeNameToOrder.get(node.name)!];
    if (selfLife == null) {
      // If nodeToOrder does not contain the node, it is unused or
      // unreachable in graph.
      return -1;
    }
    return selfLife;
  };

  // `liveUntil[i]` points to the last node in the `orderedNodes` array that
  // may depend on tensors from node `i`. It indicates that all the
  // intermediate tensors from `orderedNodes[i]` should be disposed after
  // `orderedNodes[liveUntil[i]]` is executed.
  // A node lives long enough to pass on its tensors to its children.
  // It lives until at least `max(node's position, children's positions)`.
  const liveUntilOrders = orderedNodes.map((node, nodeOrder) => {
    return node.children.map(getSelfLifeSpan)
        .reduce((a, b) => Math.max(a, b), selfLifespans[nodeOrder]);
  });

  // liveUntilMap:
  // - Key: Name of a node `x`
  // - Values: All nodes whose intermediate tensors should be disposed
  //           after `x` is executed.
  const liveUntilMap = new Map<string, Node[]>();
  for (let nodeOrder = 0; nodeOrder < orderedNodes.length; ++nodeOrder) {
    const liveUntilOrder = liveUntilOrders[nodeOrder];
    if (liveUntilOrder === INF_LIFE) {
      continue;
    }
    const node = orderedNodes[nodeOrder];
    const liveUntilNode = orderedNodes[liveUntilOrder];
    if (!liveUntilMap.has(liveUntilNode.name)) {
      liveUntilMap.set(liveUntilNode.name, []);
    }
    liveUntilMap.get(liveUntilNode.name)!.push(node);
  }
  return liveUntilMap;
}

const CONTROL_FLOW_OPS = new Set([
  'Switch', 'Merge', 'Enter', 'Exit', 'NextIteration', 'StatelessIf',
  'StatelessWhile', 'if', 'While'
]);
const DYNAMIC_SHAPE_OPS = new Set([
  'NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'NonMaxSuppressionV5', 'Where'
]);
const HASH_TABLE_OPS = new Set([
  'HashTable', 'HashTableV2', 'LookupTableImport', 'LookupTableImportV2',
  'LookupTableFind', 'LookupTableFindV2', 'LookupTableSize', 'LookupTableSizeV2'
]);

export function isControlFlow(node: Node) {
  return CONTROL_FLOW_OPS.has(node.op);
}

export function isDynamicShape(node: Node) {
  return DYNAMIC_SHAPE_OPS.has(node.op);
}

export function isHashTable(node: Node) {
  return HASH_TABLE_OPS.has(node.op);
}
