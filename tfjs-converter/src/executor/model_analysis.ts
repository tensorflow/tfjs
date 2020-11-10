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
      Object.keys(inputs).map(name => parseNodeName(name)[0]);

  let initNodeNames: string[] = [];
  if (initNodes != null) {
    initNodeNames = initNodes.map(node => parseNodeName(node.name)[0]);
  }

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
    if (inputNodeNames.indexOf(node.name) !== -1) {
      continue;
    }
    // This node is a dead end since it doesn't have any inputs.
    if (initNodeNames.indexOf(node.name) !== -1) {
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
  const frontier: Node[] = [];
  const inputNodes = Object.keys(inputs)
                         .map(name => parseNodeName(name)[0])
                         .map(name => graph.nodes[name]);
  const initNodes = graph.initNodes;

  inputNodes.forEach(input => {
    if (usedNodes.has(input.name)) {
      frontier.push(input);
    }
  });
  graph.weights.forEach(weight => {
    if (usedNodes.has(weight.name)) {
      frontier.push(weight);
    }
  });
  if (initNodes != null) {
    initNodes.forEach(node => {
      if (usedNodes.has(node.name)) {
        frontier.push(node);
      }
    });
  }
  const seen = new Set<string>();
  const orderedNodes: Node[] = [];
  while (frontier.length > 0) {
    const node = frontier.pop();
    seen.add(node.name);
    if (!weightMap[node.name]) {
      orderedNodes.push(node);
    }
    node.children.forEach(child => {
      if (!seen.has(child.name) && usedNodes.has(child.name) &&
          child.inputs.every(input => seen.has(input.name))) {
        frontier.push(child);
      }
    });
  }
  return orderedNodes;
}

const CONTROL_FLOW_OPS = [
  'Switch', 'Merge', 'Enter', 'Exit', 'NextIteration', 'StatelessIf',
  'StatelessWhile', 'if', 'While'
];
const DYNAMIC_SHAPE_OPS = [
  'NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'NonMaxSuppressionV5', 'Where'
];
const HASH_TABLE_OPS = [
  'HashTable', 'HashTableV2', 'LookupTableImport', 'LookupTableImportV2',
  'LookupTableFind', 'LookupTableFindV2'
];

export function isControlFlow(node: Node) {
  return CONTROL_FLOW_OPS.indexOf(node.op) >= 0;
}

export function isDynamicShape(node: Node) {
  return DYNAMIC_SHAPE_OPS.indexOf(node.op) >= 0;
}

export function isHashTable(node: Node) {
  return HASH_TABLE_OPS.indexOf(node.op) >= 0;
}
