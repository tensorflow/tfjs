/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {clone, Tensor, util} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {ResourceManager} from '../../executor/resource_manager';
import {Node, ValueType} from '../types';

export function getParamValue(
    paramName: string, node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext, resourceManager?: ResourceManager): ValueType {
  const inputParam = node.inputParams[paramName];
  if (inputParam && inputParam.inputIndexStart !== undefined) {
    const start = inputParam.inputIndexStart;
    const end = inputParam.inputIndexEnd === 0 ?
        undefined :
        (inputParam.inputIndexEnd === undefined ? start + 1 :
                                                  inputParam.inputIndexEnd);
    const shiftedStart = start < 0 ? node.inputNames.length + start : start;
    if (inputParam.type === 'tensor') {
      return getTensor(
          node.inputNames[shiftedStart], tensorMap, context, resourceManager);
    }
    if (inputParam.type === 'tensors') {
      // TODO(mattSoulanille): This filters out NoOp nodes during execution, but
      // these should really never be in the execution graph in the first place.
      // They're necessary for ordering the graph, but should not be visible
      // during execution. Perhaps have different sets of children, one for
      // control dependencies and another for real dependencies.
      const inputs = node.inputs.slice(start, end);
      const inputNames = node.inputNames.slice(start, end)
        .filter((_name, index) => inputs[index]?.op !== 'NoOp');

      return inputNames.map(
          name => getTensor(name, tensorMap, context, resourceManager));
    }
    const tensor = getTensor(
        node.inputNames[shiftedStart], tensorMap, context, resourceManager);
    const data = tensor.dataSync();
    return inputParam.type === 'number' ?
        data[0] :
        util.toNestedArray(tensor.shape, data);
  }
  const attrParam = node.attrParams[paramName];
  return attrParam && attrParam.value;
}

/**
 * Retrieve the tensor from tensorsMap based on input name.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
export function getTensor(
    name: string, tensorsMap: NamedTensorsMap, context: ExecutionContext,
    resourceManager?: ResourceManager): Tensor {
  const [nodeName, index] = parseNodeName(name, context);

  if (resourceManager != null) {
    const tensor = resourceManager.getHashTableHandleByName(nodeName);
    if (tensor != null) {
      return tensor;
    }
  }

  const contextId = context.currentContextIds.find(contextId => {
    return !!tensorsMap[getNodeNameWithContextId(nodeName, contextId)];
  });

  return contextId !== undefined ?
      tensorsMap[getNodeNameWithContextId(nodeName, contextId)][index] :
      undefined;
}

/**
 * Retrieve the tensors based on input name for current context.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 */
export function getTensorsForCurrentContext(
    name: string, tensorsMap: NamedTensorsMap,
    context: ExecutionContext): Tensor[] {
  return tensorsMap[getNodeNameWithContextId(name, context.currentContextId)];
}

/**
 * Returns the node name, outputName and index from the Node input name.
 * @param inputName The input name of the node, in format of
 * node_name:output_index, i.e. MatMul:0, if the output_index is not set, it is
 * default to 0.
 * If the input name contains output name i.e. StringSplit:indices:0, it will
 * return ['StringSplit', 0, 'indices'].
 */
export function getNodeNameAndIndex(
    inputName: string, context?: ExecutionContext): [string, number, string] {
  const [nodeName, index, outputName] = parseNodeName(inputName, context);

  return [
    getNodeNameWithContextId(nodeName, context && context.currentContextId),
    index, outputName
  ];
}

function getNodeNameWithContextId(name: string, contextId?: string): string {
  return !!contextId ? `${name}-${contextId}` : name;
}

export function parseNodeName(
    name: string, context?: ExecutionContext): [string, number, string?] {
  if (name === '') {
    return ['', 0, undefined];
  }

  const isCacheEnabled = context != null && context.parseNodeNameCache != null;
  if (isCacheEnabled) {
    const cachedResult = context.parseNodeNameCache.get(name);
    if (cachedResult != null) {
      return cachedResult;
    }
  }
  const parts = name.split(':');
  let result: [string, number, string?];
  if (parts.length === 1) {
    result = [name, 0, undefined];
  } else {
    const nodeName = parts[0];
    const outputName = parts.length === 3 ? parts[1] : undefined;
    const index = Number(parts[parts.length - 1]);
    result = [nodeName, index, outputName];
  }
  if (isCacheEnabled) {
    context.parseNodeNameCache.set(name, result);
  }
  return result;
}

export function split(arr: number[], size: number) {
  const res = [];
  for (let i = 0; i < arr.length; i += size) {
    res.push(arr.slice(i, i + size));
  }
  return res;
}
export function getPadding(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): ValueType {
  let pad = getParamValue('pad', node, tensorMap, context);
  if (pad === 'explicit') {
    // This is 1d array, we need to convert it to 2d array
    pad = getParamValue('explicitPaddings', node, tensorMap, context);
    const explicitPadding: [
      [number, number], [number, number], [number, number], [number, number]
    ] = [[0, 0], [0, 0], [0, 0], [0, 0]];
    for (let i = 0; i < 4; i++) {
      explicitPadding[i][0] = (pad as number[])[i * 2];
      explicitPadding[i][1] = (pad as number[])[i * 2 + 1];
    }
    return explicitPadding;
  }
  return pad;
}

/**
 *  Reuse the tensor if it is marked as keep, otherwise clone the tensor to
 *  avoid disposal. This is important for TensorArray and TensorList ops, since
 *  internally they use a tensor as the id for TensorArray and TensorList, and
 * to simplify lookup, they also use Tensor.id as the key to the internal map.
 * These id tensors have been marked as kept in the backend, we need avoid clone
 * them in order to create new Tensor.id.
 * @param tensor
 */
export function cloneTensor(tensor: Tensor): Tensor {
  return tensor.kept ? tensor : clone(tensor);
}
