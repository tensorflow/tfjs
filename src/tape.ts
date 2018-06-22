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

import {Tensor} from './tensor';
import {NamedTensorMap} from './tensor_types';
import * as util from './util';

export interface TapeNode {
  id: number;
  name: string;
  output: Tensor;
  inputs: NamedTensorMap;
  // Optional params, defined only for ops with gradient impl.
  gradient?: (dy: Tensor|NamedTensorMap) => NamedGradientMap;
}

export type NamedGradientMap = {
  [inputName: string]: () => Tensor;
};

/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 * @param tape The tape elements to filter.
 * @param xs The input Tensors.
 * @param y The output Tensor.
 */
export function getFilteredNodesXToY(
    tape: TapeNode[], xs: Tensor[], y: Tensor): TapeNode[] {
  // Forward pass to compute all the nodes and Tensors that are transitively a
  // function of x.
  const tensorsFromX: {[tensorId: number]: boolean} = {};
  const nodesFromX: {[nodeId: number]: boolean} = {};
  for (let i = 0; i < xs.length; i++) {
    tensorsFromX[xs[i].id] = true;
  }

  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];
    const nodeInputs = node.inputs;
    for (const inputName in nodeInputs) {
      const input = nodeInputs[inputName];

      let anyInputFromX = false;
      for (let j = 0; j < xs.length; j++) {
        if (tensorsFromX[input.id]) {
          tensorsFromX[node.output.id] = true;
          anyInputFromX = true;
          nodesFromX[node.id] = true;
          break;
        }
      }

      if (anyInputFromX) {
        break;
      }
    }
  }

  // Backwards pass to find all of the nodes and Tensors that lead to y.
  const tensorsLeadToY: {[tensorId: number]: boolean} = {};
  tensorsLeadToY[y.id] = true;
  const nodesToY: {[nodeId: number]: boolean} = {};

  for (let i = tape.length - 1; i >= 0; i--) {
    const node = tape[i];
    const nodeInputs = node.inputs;

    const outputs: Tensor[] = [];
    outputs.push(node.output);

    // If any of the outputs lead to y, mark all of the inputs as leading to y.
    for (let j = 0; j < outputs.length; j++) {
      if (tensorsLeadToY[outputs[j].id]) {
        for (const inputName in nodeInputs) {
          tensorsLeadToY[nodeInputs[inputName].id] = true;
          nodesToY[node.id] = true;
        }
        break;
      }
    }
  }

  // Return the paths that come from x and lead to y.
  const filteredTape: TapeNode[] = [];
  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];

    if (nodesFromX[node.id] && nodesToY[node.id]) {
      // Prune the inputs from the node that aren't a function of x.
      const prunedInputs: {[inputName: string]: Tensor} = {};
      for (const inputName in node.inputs) {
        const nodeInput = node.inputs[inputName];
        if (tensorsFromX[nodeInput.id]) {
          prunedInputs[inputName] = nodeInput;
        }
      }

      // Copy the node and overwrite inputsAndArgs to the pruned version.
      const prunedNode = Object.assign({}, node) as TapeNode;
      prunedNode.inputs = prunedInputs;
      prunedNode.output = node.output;

      filteredTape.push(prunedNode);
    }
  }

  return filteredTape;
}

/**
 * Backpropagate gradients through the filtered TapeNodes.
 * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
export function backpropagateGradients(
    tensorAccumulatedGradientMap: {[tensorId: number]: Tensor},
    filteredTape: TapeNode[]) {
  // Walk the tape backwards and keep a map of Tensor to its gradient.
  for (let i = filteredTape.length - 1; i >= 0; i--) {
    const node = filteredTape[i];

    const dy = tensorAccumulatedGradientMap[node.output.id];

    if (node.gradient == null) {
      throw new Error(
          `Cannot compute gradient: gradient function not found ` +
          `for ${node.name}.`);
    }

    // Backprop dy through this node and accumulate gradients over the inputs.
    const inputGradients = node.gradient(dy);
    for (const inputName in node.inputs) {
      if (!(inputName in inputGradients)) {
        throw new Error(
            `Cannot backprop through input ${inputName}. ` +
            `Available gradients found: ${Object.keys(inputGradients)}.`);
      }

      // Call the gradient function.
      const dx = inputGradients[inputName]();
      const x = node.inputs[inputName];
      if (!util.arraysEqual(dx.shape, x.shape)) {
        throw new Error(
            `Error in gradient for op ${node.name}. The gradient of input ` +
            `'${inputName}' has shape '${dx.shape}', which does not match ` +
            `the shape of the input '${x.shape}'`);
      }

      if (tensorAccumulatedGradientMap[x.id] == null) {
        tensorAccumulatedGradientMap[x.id] = dx;
      } else {
        const curGradient = tensorAccumulatedGradientMap[x.id];
        tensorAccumulatedGradientMap[x.id] = curGradient.add(dx);
        curGradient.dispose();
      }
    }
  }
}
