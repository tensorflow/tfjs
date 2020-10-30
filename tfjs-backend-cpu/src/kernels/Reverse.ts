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

import {KernelConfig, KernelFunc, Reverse, ReverseAttrs, ReverseInputs, TensorBuffer, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {identity} from './Identity';

export function reverse(
    args:
        {inputs: ReverseInputs, backend: MathBackendCPU, attrs: ReverseAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dims} = attrs;

  assertNotComplex(x, 'reverse');

  const xRank = x.shape.length;

  const $dims = util.parseAxisParam(dims, x.shape);
  if (xRank === 0) {
    return identity({inputs: {x}, backend});
  }

  const buffer = new TensorBuffer(x.shape, x.dtype);
  const xBuf = backend.bufferSync(x);

  for (let i = 0; i < buffer.size; i++) {
    const outLoc = buffer.indexToLoc(i);
    const inLoc = outLoc.slice();
    $dims.forEach(d => inLoc[d] = x.shape[d] - 1 - inLoc[d]);
    buffer.set(xBuf.get(...inLoc), ...outLoc);
  }

  return backend.makeTensorInfo(buffer.shape, buffer.dtype, buffer.values);
}

export const reverseConfig: KernelConfig = {
  kernelName: Reverse,
  backendName: 'cpu',
  kernelFunc: reverse as {} as KernelFunc
};
