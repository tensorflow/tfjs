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

import {KernelConfig, KernelFunc, LRNGrad, LRNGradAttrs, LRNGradInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function lRNGrad(
    args:
        {inputs: LRNGradInputs, backend: MathBackendCPU, attrs: LRNGradAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, y, dy} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  assertNotComplex(dy, 'LRNGrad');

  const dySize = util.sizeFromShape(dy.shape);

  const channels = dy.shape[3];
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  const xValues = backend.data.get(x.dataId).values as TypedArray;
  const yValues = backend.data.get(y.dataId).values as TypedArray;
  const result = new Float32Array(dySize);
  const size = dySize;

  for (let offset = 0; offset < size; offset++) {
    const currentChannel = offset % channels;
    const depthBegin =
        (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
    const depthEnd = (offset - currentChannel) +
        Math.min(channels, currentChannel + depthRadius + 1);

    let norm = 0;
    for (let k = depthBegin; k < depthEnd; k++) {
      norm += Math.pow(xValues[k], 2);
    }
    norm = alpha * norm + bias;

    for (let k = depthBegin; k < depthEnd; k++) {
      let dyi = -2 * alpha * beta * xValues[k] * yValues[offset] / norm;
      if (offset === k) {
        dyi += Math.pow(norm, -beta);
      }
      dyi *= dyValues[offset];
      result[k] += dyi;
    }
  }

  return backend.makeTensorInfo(dy.shape, x.dtype, result);
}

export const lRNGradConfig: KernelConfig = {
  kernelName: LRNGrad,
  backendName: 'cpu',
  kernelFunc: lRNGrad as {} as KernelFunc
};
