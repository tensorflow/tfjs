/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {NamedTensorMap} from '../../data/index';
import {Node, ValueType} from '../index';

export function getParamValue(
    paramName: string, node: Node, tensorMap: NamedTensorMap): ValueType {
  const param = node.params[paramName];
  if (param && param.inputIndex !== undefined) {
    if (param.type === 'tensor') {
      return tensorMap[node.inputNames[param.inputIndex]];
    }
    if (param.type === 'tensors') {
      const inputs = param.inputIndex === 0 ?
          node.inputNames.slice(param.inputIndex, -param.inputParamLength) :
          node.inputNames.splice(param.inputIndex);
      return inputs.map(name => tensorMap[name]);
    }
    const data = Array.prototype.slice.call(
        tensorMap[node.inputNames.slice(param.inputIndex)[0]].dataSync());
    return param.type === 'number' ? data[0] : data;
  }
  return param && param.value;
}
