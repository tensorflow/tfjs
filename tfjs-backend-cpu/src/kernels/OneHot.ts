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

import {KernelConfig, KernelFunc, OneHot, OneHotAttrs, OneHotInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function oneHot(
    args: {inputs: OneHotInputs, backend: MathBackendCPU, attrs: OneHotAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {indices} = inputs;
  const {depth, onValue, offValue} = attrs;

  assertNotComplex(indices, 'oneHot');

  const indicesSize = util.sizeFromShape(indices.shape);

  const res = new Float32Array(indicesSize * depth);
  res.fill(offValue);
  const indicesVal = backend.data.get(indices.dataId).values as TypedArray;

  for (let event = 0; event < indicesSize; ++event) {
    if (indicesVal[event] >= 0 && indicesVal[event] < depth) {
      res[event * depth + indicesVal[event]] = onValue;
    }
  }

  return backend.makeTensorInfo([...indices.shape, depth], 'int32', res);
}

export const oneHotConfig: KernelConfig = {
  kernelName: OneHot,
  backendName: 'cpu',
  kernelFunc: oneHot as {} as KernelFunc
};
