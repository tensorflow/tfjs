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

import {KernelConfig, KernelFunc, Reshape, ReshapeAttrs, ReshapeInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function reshape(
    args:
        {inputs: ReshapeInputs, backend: MathBackendCPU, attrs: ReshapeAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {shape} = attrs;

  const xSize = util.sizeFromShape(x.shape);
  const $shape = util.inferFromImplicitShape(shape, xSize);
  const $xSize = util.sizeFromShape($shape);

  util.assert(
      xSize === $xSize,
      () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
          `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
          `shape must have the same number of elements.`);

  backend.incRef(x.dataId);

  const xData = backend.data.get(x.dataId);

  if (xData.complexTensorInfos != null) {
    const real = xData.complexTensorInfos.real;
    const imag = xData.complexTensorInfos.imag;

    real.shape = $shape;
    imag.shape = $shape;
  }

  return {dataId: x.dataId, shape: $shape, dtype: x.dtype};
}

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'cpu',
  kernelFunc: reshape as {} as KernelFunc
};
