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
    graph: Graph, weightMap: NamedTensorsMap,
    executionInfo: ExecutionInfo): Node[] {
  const {usedNodes, inputs} = executionInfo;
  const inputNodes = Object.keys(inputs)
                         .map(name => parseNodeName(name)[0])
                         .map(name => graph.nodes[name]);
  const initNodes = graph.initNodes;

  const isUsed = (node: Node|string) =>
      usedNodes.has(typeof node === 'string' ? node : node.name);
  const frontier: Node[] = [
    ...inputNodes,
    ...graph.weights,
    ...(initNodes || []),
  ].filter(isUsed).filter((node) => {
    // If predefined node has any inputs, skips it and waits until topological
    // sort resolves its parents to add it to frontier.
    return node.inputs.length === 0;
  });

  const orderedNodes: Node[] = [];
  const processedNodeNames = new Set<string>();
  const isProcessed = (node: Node|string) =>
      processedNodeNames.has(typeof node === 'string' ? node : node.name);

  // TODO: Improve performance of the topological sort implementation.
  while (frontier.length > 0) {
    const node = frontier.pop();
    if (isProcessed(node)) {
      continue;
    }
    processedNodeNames.add(node.name);
    orderedNodes.push(node);
    for (const child of node.children.filter(isUsed)) {
      if (isProcessed(child)) {
        throw new Error(
            'Topological Sort Error: child should not be processed.');
      }
      if (child.inputs.map((node) => node.name).every(isProcessed)) {
        frontier.push(child);
      }
    }
  }

  // Validates node orders
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));
  const allNodes = [
    ...orderedNodes, ...inputNodes, ...graph.weights, ...(initNodes || [])
  ].filter(isUsed);
  for (const node of allNodes) {
    for (const child of node.children.filter(isUsed)) {
      if (!nodeNameToOrder.has(child.name)) {
        throw new Error('Topological Sort Error: child is unreachable.');
      }
      if (nodeNameToOrder.get(node.name) > nodeNameToOrder.get(child.name)) {
        throw new Error(
            'Topological Sort Error: node has greater order than its child.');
      }
    }
    for (const input of node.inputs) {
      if (!nodeNameToOrder.has(input.name)) {
        throw new Error('Topological Sort Error: input is unreachable.');
      }
      if (nodeNameToOrder.get(input.name) > nodeNameToOrder.get(node.name)) {
        throw new Error(
            'Topological Sort Error: node has smaller order than its input.');
      }
    }
  }

  return orderedNodes;
}

/**
 * Given the execution info, return a map from node name to the disposable node
 * name list after its execution.
 *
 * @returns A map from node name to disposable node names after its
 *     execution. That is, for a node `x`, `nodeLiveUntilMap[x]` indicates all
 *     nodes which their intermediate tensors should be disposed after `x` being
 *     executed.
 */
export function getNodeLiveUntilMap(orderedNodes: Node[]):
    Map<string, string[]> {
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));

  const INF_LIFE = Number.MAX_SAFE_INTEGER;
  // Make control flow nodes (and consequently their direct parents)
  // live forever since they're tricky to track correctly.
  const selfLifespans = orderedNodes.map(
      (node, nodeOrder) => isControlFlow(node) ? INF_LIFE : nodeOrder);
  const getSelfLifeSpan =
      (node: Node) => {
        const selfLife = selfLifespans[nodeNameToOrder.get(node.name)!];
        if (selfLife == null) {
          // If nodeToOrder does not contain the node, it is unused or
          // unreachable in graph.
          return -1;
        }
        return selfLife;
      }

  // `liveUntil[i]` points to the last node in the `orderedNodes` array that
  // may depend on tensors from node `i`. It indicates that all the intermediate
  // tensors from `orderedNodes[i]` should be disposed after
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
  const liveUntilMap = new Map<string, string[]>();
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
    liveUntilMap.get(liveUntilNode.name)!.push(node.name);
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
