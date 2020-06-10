/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

/**
 * Create a list of input tensors.
 * @param inputsData An array with each element being the value to create a
 *    tensor.
 * @param inputsShapes An array with each element being the shape to create a
 *    tensor.
 */
export function createInputTensors(
    inputsData: tfc.TypedArray[], inputsShapes: number[][],
    inputDtypes?: tfc.DataType[], inputNames?: string[]): tfc.Tensor[]|
    tfc.NamedTensorMap {
  const xs: tfc.Tensor[] = [];
  for (let i = 0; i < inputsData.length; i++) {
    const input = tfc.tensor(
        inputsData[i], inputsShapes[i],
        inputDtypes ? inputDtypes[i] : 'float32');
    xs.push(input);
  }
  if (inputNames) {
    return inputNames.reduce((map: tfc.NamedTensorMap, name, index) => {
      map[name] = xs[index];
      return map;
    }, {});
  }
  return xs;
}
