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

import {backend_util, KernelConfig, KernelFunc, SparseToDense, SparseToDenseAttrs, SparseToDenseInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {scatterImpl} from './Scatter_impl';

export function sparseToDense(args: {
  inputs: SparseToDenseInputs,
  backend: MathBackendCPU,
  attrs: SparseToDenseAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sparseIndices, sparseValues, defaultValue} = inputs;
  const {outputShape} = attrs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
  const sumDupeIndices = false;

  const indicesBuf = backend.bufferSync(sparseIndices);
  const updatesBuf = backend.bufferSync(sparseValues);
  const $defaultValue =
      backend.data.get(defaultValue.dataId).values[0] as number;

  const outBuf = scatterImpl(
      indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates,
      sliceRank, strides, $defaultValue, sumDupeIndices);

  return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'cpu',
  kernelFunc: sparseToDense as {} as KernelFunc
};
