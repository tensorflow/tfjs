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

import {backend_util, KernelConfig, KernelFunc, ScatterNd, ScatterNdAttrs, ScatterNdInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {scatterImpl} from './Scatter_impl';

export function scatterNd(args: {
  inputs: ScatterNdInputs,
  backend: MathBackendCPU,
  attrs: ScatterNdAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {indices, updates} = inputs;
  const {shape} = attrs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(updates, indices, shape);
  const sumDupeIndices = true;

  const indicesBuf = backend.bufferSync(indices);
  const updatesBuf = backend.bufferSync(updates);

  const outBuf = scatterImpl(
      indicesBuf, updatesBuf, shape, outputSize, sliceSize, numUpdates,
      sliceRank, strides, 0 /* defaultValue */, sumDupeIndices);

  return backend.makeTensorInfo(shape, outBuf.dtype, outBuf.values);
}

export const scatterNdConfig: KernelConfig = {
  kernelName: ScatterNd,
  backendName: 'cpu',
  kernelFunc: scatterNd as {} as KernelFunc
};
