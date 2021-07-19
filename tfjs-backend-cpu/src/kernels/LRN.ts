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

import {KernelConfig, KernelFunc, LRN, LRNAttrs, LRNInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function lRN(
    args: {inputs: LRNInputs, backend: MathBackendCPU, attrs: LRNAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  assertNotComplex(x, 'LRN');

  const channels = x.shape[3];
  const maxD = channels - 1;
  const xValues = backend.data.get(x.dataId).values as TypedArray;
  const size = util.sizeFromShape(x.shape);
  const result = new Float32Array(size);

  function sumAcrossChannels(offset: number) {
    const currentChannel = offset % channels;
    let beginSumOffset =
        offset - currentChannel + Math.max(0, currentChannel - depthRadius);
    const endSumOffset =
        offset - currentChannel + Math.min(currentChannel + depthRadius, maxD);

    let sum = 0.0;
    for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
      const z = xValues[beginSumOffset];
      sum += z * z;
    }
    return sum;
  }

  for (let offset = 0; offset < size; offset++) {
    const sum = sumAcrossChannels(offset);
    const val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
    result[offset] = val;
  }

  return backend.makeTensorInfo(x.shape, x.dtype, result);
}

export const lRNConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'cpu',
  kernelFunc: lRN as {} as KernelFunc
};
