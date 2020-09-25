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
  missingInputs: Node[];
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
    inputs: NamedTensorMap, outputs: Node[],
    weightMap: NamedTensorsMap): ExecutionInfo {
  const usedNodes = new Set<string>();
  const missingInputs: Node[] = [];
  let dynamicNode: Node = null;
  let syncInputs: string[] = null;

  // Start with the outputs, going backwards and find all the nodes that are
  // needed to compute those outputs.
  const seen = new Set<string>();
  const inputNodeNames =
      Object.keys(inputs).map(name => parseNodeName(name)[0]);
  const frontier = [...outputs];
  while (frontier.length > 0) {
    const node = frontier.pop();
    if (isControlFlow(node) || isDynamicShape(node)) {
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
    if (node.inputs.length === 0) {
      missingInputs.push(node);
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

/**
 * Given the outputs, return a list of nodes in topological order that
 * need to be executed to compute the outputs.
 *
 * This method is used for building initializer subgraph. The differences
 * between this method and `getNodesInTopologicalOrder` is that the input nodes
 * are not required to be tensors, they can be ops.
 *
 * @param outputs: Output nodes that the algorithm use to trace back to input
 *     nodes and then calculate execution order from the input nodes.
 */
export function getGraphInTopologicalOrder(outputs: Node[]): Node[] {
  const inputs = findInputs(outputs);
  const frontier = [...inputs];
  const seen = new Set<string>();
  const orderedNodes: Node[] = [];

  while (frontier.length > 0) {
    const top = frontier.pop();
    seen.add(top.name);
    orderedNodes.push(top);
    top.children.forEach(child => {
      if (!seen.has(child.name) &&
          child.inputs.every(input => seen.has(input.name))) {
        frontier.push(child);
      }
    });
  }

  return orderedNodes;
}

// Output nodes that the algorithm use to trace back to input nodes.
function findInputs(outputs: Node[]): Node[] {
  const frontier = [...outputs];
  const inputs = [];
  const seen = new Set<string>();

  while (frontier.length > 0) {
    const top = frontier.pop();

    if (!seen.has(top.name)) {
      if (top.inputs.length > 0) {
        frontier.push(...top.inputs);
      } else {
        inputs.push(top);
      }
      seen.add(top.name);
    }
  }

  return inputs;
}

const CONTROL_FLOW_OPS = [
  'Switch', 'Merge', 'Enter', 'Exit', 'NextIteration', 'StatelessIf',
  'StatelessWhile', 'if', 'While'
];
const DYNAMIC_SHAPE_OPS = [
  'NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'NonMaxSuppressionV5', 'Where'
];

export function isControlFlow(node: Node) {
  return CONTROL_FLOW_OPS.indexOf(node.op) >= 0;
}

export function isDynamicShape(node: Node) {
  return DYNAMIC_SHAPE_OPS.indexOf(node.op) >= 0;
}
