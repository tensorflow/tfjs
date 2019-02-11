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

import * as tfc from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {Node, ValueType} from '../types';

export function getParamValue(
    paramName: string, node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): ValueType {
  const inputParam = node.inputParams[paramName];
  if (inputParam && inputParam.inputIndexStart !== undefined) {
    const start = inputParam.inputIndexStart;
    const end = inputParam.inputIndexEnd === 0 ?
        undefined :
        (inputParam.inputIndexEnd === undefined ? start + 1 :
                                                  inputParam.inputIndexEnd);
    if (inputParam.type === 'tensor') {
      return getTensor(
          node.inputNames[inputParam.inputIndexStart], tensorMap, context);
    }
    if (inputParam.type === 'tensors') {
      const inputs = node.inputNames.slice(start, end);

      return inputs.map(name => getTensor(name, tensorMap, context));
    }
    const data = Array.prototype.slice.call(
        getTensor(node.inputNames.slice(start)[0], tensorMap, context)
            .dataSync());
    return inputParam.type === 'number' ? data[0] : data;
  }
  const attrParam = node.attrParams[paramName];
  return attrParam && attrParam.value;
}

/**
 * Retrieve the tensor based on input name by extracting the node name and
 * output index information.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 */
export function getTensor(
    name: string, tensorsMap: NamedTensorsMap,
    context: ExecutionContext): tfc.Tensor {
  const [nodeName, index] = parseNodeName(name);
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
export function getTensorsForCurrentContenxt(
    name: string, tensorsMap: NamedTensorsMap,
    context: ExecutionContext): tfc.Tensor[] {
  return tensorsMap[getNodeNameWithContextId(name, context.currentContextId)];
}

/**
 * Returns the node name and index from the Node input name.
 * @param inputName The input name of the node, in format of
 * node_name:output_index, i.e. MatMul:0, if the output_index is not set, it is
 * default to 0.
 */
export function getNodeNameAndIndex(
    inputName: string, context?: ExecutionContext): [string, number] {
  const [nodeName, index] = parseNodeName(inputName);

  return [
    getNodeNameWithContextId(nodeName, context && context.currentContextId),
    index
  ];
}

function getNodeNameWithContextId(name: string, contextId?: string): string {
  return !!contextId ? `${name}-${contextId}` : name;
}

export function parseNodeName(name: string): [string, number] {
  const index = name.lastIndexOf(':');
  if (index === -1) return [name, 0];

  const nodeName = name.substring(0, index);
  return [nodeName, Number(name.substring(index + 1))];
}

export function split(arr: number[], size: number) {
  const res = [];
  for (let i = 0; i < arr.length; i += size) {
    res.push(arr.slice(i, i + size));
  }
  return res;
}
