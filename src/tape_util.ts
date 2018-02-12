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

import * as util from './util';
import {Tensor} from './tensor';
import {NamedTensorMap, RegularArray} from './types';

// tslint:disable-next-line:max-line-length
import {Tape, TapeNode, TapeNodeInputConfig, TapeNodeOutput} from './tape_types';

/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 * @param tape The tape elements to filter.
 * @param xx The input Tensors.
 * @param y The output Tensor.
 */
export function getFilteredNodesXToY(
    tape: Tape, xs: Tensor[], y: Tensor): Tape {
  // Forward pass to compute all the nodes and Tensors that are transitively a
  // function of x.
  const tensorsFromX: {[tensorId: number]: boolean} = {};
  const nodesFromX: {[nodeId: number]: boolean} = {};
  for (let i = 0; i < xs.length; i++) {
    tensorsFromX[xs[i].id] = true;
  }

  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];
    const nodeInputs = node.inputAndArgs.inputs;

    for (const inputName in nodeInputs) {
      const input = nodeInputs[inputName];

      let anyInputFromX = false;
      for (let j = 0; j < xs.length; j++) {
        if (tensorsFromX[input.id]) {
          if (node.output instanceof Tensor) {
            tensorsFromX[node.output.id] = true;
          } else {
            const keys = Object.keys(node.output);
            for (const key of keys) {
              tensorsFromX[node.output[key].id] = true;
            }
          }
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
    const nodeInputs = node.inputAndArgs.inputs;

    const outputs: Tensor[] = [];
    if (node.output instanceof Tensor) {
      outputs.push(node.output);
    } else {
      const keys = Object.keys(node.output);
      for (const key of keys) {
        outputs.push(node.output[key]);
      }
    }

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
  const filteredTape: Tape = [];
  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];

    if (nodesFromX[node.id] && nodesToY[node.id]) {
      // Prune the inputs from the node that aren't a function of x.
      const prunedInputs: {[inputName: string]: Tensor} = {};
      for (const inputName in node.inputAndArgs.inputs) {
        const nodeInput = node.inputAndArgs.inputs[inputName];
        if (tensorsFromX[nodeInput.id]) {
          prunedInputs[inputName] = nodeInput;
        }
      }

      let prunedOutputs: Tensor|{[outputName: string]: Tensor};
      if (node.output instanceof Tensor) {
        // Nothing to prune if the output is just a single Tensor since the
        // node would have been pruned.
        prunedOutputs = node.output;
      } else {
        // Prune the outputs from the node that don't lead to y.
        prunedOutputs = {};
        for (const outputName in node.output) {
          const output = node.output[outputName];
          if (tensorsLeadToY[output.id]) {
            prunedOutputs[outputName] = node.output[outputName];
          }
        }
      }

      // Copy the node and overwrite inputsAndArgs to the pruned version.
      const prunedNode = Object.assign({}, node) as TapeNode<TapeNodeOutput>;
      prunedNode.inputAndArgs = {inputs: prunedInputs};
      prunedNode.output = prunedOutputs;

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
    filteredTape: Tape) {
  // Walk the tape backwards and keep a map of Tensor to its gradient.
  for (let i = filteredTape.length - 1; i >= 0; i--) {
    const node = filteredTape[i];

    let dy: Tensor|NamedTensorMap;
    if (node.output instanceof Tensor) {
      dy = tensorAccumulatedGradientMap[node.output.id];
    } else {
      dy = {};
      const keys = Object.keys(node.output);
      for (const key of keys) {
        dy[key] = tensorAccumulatedGradientMap[node.output[key].id];
      }
    }

    if (node.gradient == null) {
      throw new Error(
          `Cannot compute gradient: gradient function not found ` +
          `for ${node.name}.`);
    }

    // Backprop dy through this node and accumulate gradients over the inputs.
    const inputGradients = node.gradient(dy, node.output);
    for (const inputName in node.inputAndArgs.inputs) {
      if (!(inputName in inputGradients)) {
        throw new Error(
            `Cannot backprop through input ${inputName}. ` +
            `Available gradients found: ${Object.keys(inputGradients)}.`);
      }

      // Call the gradient function.
      const dx = inputGradients[inputName]();
      const x = node.inputAndArgs.inputs[inputName];
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

export type ScopeResultImmediate =
    void|Tensor|RegularArray<Tensor>|{[key: string]: Tensor | Tensor[]};
/** @docalias Tensor|Tensor[]|{[key: string]: Tensor}|void */
export type ScopeResult = ScopeResultImmediate|Promise<ScopeResultImmediate>;
/** @docalias Function */
export type ScopeFn<T extends ScopeResult> = () => T;

export function extractTensorsFromScopeResult(result: ScopeResultImmediate):
    Tensor[] {
  if (result == null) {
    return [];
  }
  if (result instanceof Tensor) {
    return [result];
  }

  const list: Tensor[] = [];
  const resultObj = result as {[key: string]: Tensor};
  // Iteration over keys works also for arrays.
  for (const k in resultObj) {
    const sublist = util.flatten(resultObj[k]).filter(x => x instanceof Tensor);
    list.push(...sublist);
  }
  return list;
}

export function stripUndefinedInputsFromInputConfig(
    config: TapeNodeInputConfig): TapeNodeInputConfig {
  const keys = Object.keys(config.inputs);
  keys.forEach(key => {
    if (config.inputs[key] == null) {
      delete config.inputs[key];
    }
  });
  return config;
}
