/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, KernelConfig, KernelFunc, Rank, TensorInfo, TensorScatterUpdate, TensorScatterUpdateAttrs, TensorScatterUpdateInputs} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {scatterImpl} from './Scatter_impl';

export function tensorScatterUpdate(args: {
  inputs: TensorScatterUpdateInputs,
  backend: MathBackendCPU,
  attrs: TensorScatterUpdateAttrs
}): TensorInfo {
  const {inputs, backend} = args;
  const {tensor, indices, updates} = inputs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(updates, indices, tensor.shape);
  const sumDupeIndices = false;

  const indicesBuf = backend.bufferSync<Rank, 'int32'>(indices);
  const updatesBuf = backend.bufferSync<Rank, 'int32'|'float32'>(updates);
  const tensorBuf = backend.bufferSync<Rank, 'int32'|'float32'>(tensor);
  const outBuf = scatterImpl(
      indicesBuf, updatesBuf, tensor.shape, outputSize, sliceSize, numUpdates,
      sliceRank, strides, tensorBuf, sumDupeIndices);
  return backend.makeTensorInfo(tensor.shape, outBuf.dtype, outBuf.values);
}

export const tensorScatterUpdateConfig: KernelConfig = {
  kernelName: TensorScatterUpdate,
  backendName: 'cpu',
  kernelFunc: tensorScatterUpdate as unknown as KernelFunc
};
