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

import {ENV} from '../../environment';
import * as util from '../../util';
import {NDArray} from '../ndarray';

// tslint:disable-next-line:max-line-length
import {Tape, TapeNode, TapeNodeInputConfig, TapeNodeOutput} from './tape_types';

/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 * @param tape The tape elements to filter.
 * @param xx The input NDArrays.
 * @param y The output NDArray.
 */
export function getFilteredNodesXToY(
    tape: Tape, xs: NDArray[], y: NDArray): Tape {
  // Forward pass to compute all the nodes and NDArrays that are transitively a
  // function of x.
  const arraysFromX: {[ndarrayId: number]: boolean} = {};
  const nodesFromX: {[nodeId: number]: boolean} = {};
  for (let i = 0; i < xs.length; i++) {
    arraysFromX[xs[i].id] = true;
  }

  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];
    const nodeInputs = node.inputAndArgs.inputs;

    for (const inputName in nodeInputs) {
      const input = nodeInputs[inputName];

      let anyInputFromX = false;
      for (let j = 0; j < xs.length; j++) {
        if (arraysFromX[input.id]) {
          if (node.output instanceof NDArray) {
            arraysFromX[node.output.id] = true;
          } else {
            const keys = Object.keys(node.output);
            for (const key of keys) {
              arraysFromX[node.output[key].id] = true;
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

  // Backwards pass to find all of the nodes and NDArrays that lead to y.
  const arraysLeadToY: {[ndarrayId: number]: boolean} = {};
  arraysLeadToY[y.id] = true;
  const nodesToY: {[nodeId: number]: boolean} = {};

  for (let i = tape.length - 1; i >= 0; i--) {
    const node = tape[i];
    const nodeInputs = node.inputAndArgs.inputs;

    const outputs: NDArray[] = [];
    if (node.output instanceof NDArray) {
      outputs.push(node.output);
    } else {
      const keys = Object.keys(node.output);
      for (const key of keys) {
        outputs.push(node.output[key]);
      }
    }

    // If any of the outputs lead to y, mark all of the inputs as leading to y.
    for (let j = 0; j < outputs.length; j++) {
      if (arraysLeadToY[outputs[j].id]) {
        for (const inputName in nodeInputs) {
          arraysLeadToY[nodeInputs[inputName].id] = true;
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
      const prunedInputs: {[inputName: string]: NDArray} = {};
      for (const inputName in node.inputAndArgs.inputs) {
        const nodeInput = node.inputAndArgs.inputs[inputName];
        if (arraysFromX[nodeInput.id]) {
          prunedInputs[inputName] = nodeInput;
        }
      }

      let prunedOutputs: NDArray|{[outputName: string]: NDArray};
      if (node.output instanceof NDArray) {
        // Nothing to prune if the output is just a single NDArray since the
        // node would have been pruned.
        prunedOutputs = node.output;
      } else {
        // Prune the outputs from the node that don't lead to y.
        prunedOutputs = {};
        for (const outputName in node.output) {
          const output = node.output[outputName];
          if (arraysLeadToY[output.id]) {
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
 * @param arrayAccumulatedGradientMap A map of NDArray to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
export function backpropagateGradients(
    arrayAccumulatedGradientMap: {[ndarrayId: number]: NDArray},
    filteredTape: Tape) {
  // Walk the tape backwards and keep a map of NDArray to its gradient.
  for (let i = filteredTape.length - 1; i >= 0; i--) {
    const node = filteredTape[i];

    let dy: TapeNodeOutput;
    if (node.output instanceof NDArray) {
      dy = arrayAccumulatedGradientMap[node.output.id];
    } else {
      dy = {};
      const keys = Object.keys(node.output);
      for (const key of keys) {
        dy[key] = arrayAccumulatedGradientMap[node.output[key].id];
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

      if (arrayAccumulatedGradientMap[x.id] == null) {
        arrayAccumulatedGradientMap[x.id] = dx;
      } else {
        const curGradient = arrayAccumulatedGradientMap[x.id];
        arrayAccumulatedGradientMap[x.id] = ENV.math.add(curGradient, dx);
        curGradient.dispose();
      }
    }
  }
}

export function computeInputs(tape: Tape): {[idx: string]: NDArray} {
  const outputArrays: {[id: number]: boolean} = {};
  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];
    if (node.output instanceof NDArray) {
      outputArrays[node.output.id] = true;
    } else {
      const keys = Object.keys(node.output);
      for (const key of keys) {
        outputArrays[node.output[key].id] = true;
      }
    }
  }

  const inputArrays: {[idx: string]: NDArray} = {};
  const inputArraysSeen: {[ndarrayId: number]: boolean} = {};
  let idx = 0;
  for (let i = 0; i < tape.length; i++) {
    const node = tape[i];
    const inputs = node.inputAndArgs.inputs;

    const keys = Object.keys(inputs);
    for (const key of keys) {
      if (!outputArrays[inputs[key].id] && !inputArraysSeen[inputs[key].id]) {
        inputArrays[(idx++).toString()] = inputs[key];
        inputArraysSeen[inputs[key].id] = true;
      }
    }
  }
  return inputArrays;
}

export type ScopeResultImmediate =
    void|NDArray|NDArray[]|{[key: string]: NDArray};
export type ScopeResult = ScopeResultImmediate|Promise<ScopeResultImmediate>;

export function extractNDArraysFromScopeResult(result: ScopeResultImmediate):
    NDArray[] {
  if (result == null) {
    return [];
  }
  if (result instanceof NDArray) {
    return [result];
  }

  const list: NDArray[] = [];
  const resultObj = result as {[key: string]: NDArray};
  // Iteration over keys works also for arrays.
  for (const k in resultObj) {
    const val = resultObj[k];
    if (val instanceof NDArray) {
      list.push(val);
    }
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
