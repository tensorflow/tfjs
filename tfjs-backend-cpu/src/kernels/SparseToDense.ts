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

import {backend_util, KernelConfig, KernelFunc, Rank, SparseToDense, SparseToDenseAttrs, SparseToDenseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

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

  const indicesBuf = backend.bufferSync<Rank, 'int32'>(sparseIndices);

  let outBuf;
  switch (sparseValues.dtype) {
    case 'bool': {
      const updatesBuf = backend.bufferSync<Rank, 'bool'>(sparseValues);
      const $defaultValue =
          Boolean(backend.data.get(defaultValue.dataId).values[0]);
      outBuf = scatterImpl(
          indicesBuf, updatesBuf, outputShape, outputSize, sliceSize,
          numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
      break;
    }
    case 'float32': {
      const updatesBuf = backend.bufferSync<Rank, 'float32'>(sparseValues);
      const $defaultValue =
          backend.data.get(defaultValue.dataId).values[0] as number;
      outBuf = scatterImpl(
          indicesBuf, updatesBuf, outputShape, outputSize, sliceSize,
          numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
      break;
    }
    case 'int32': {
      const updatesBuf = backend.bufferSync<Rank, 'int32'>(sparseValues);
      const $defaultValue =
          backend.data.get(defaultValue.dataId).values[0] as number;
      outBuf = scatterImpl(
          indicesBuf, updatesBuf, outputShape, outputSize, sliceSize,
          numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
      break;
    }
    case 'string': {
      const updatesBuf = backend.bufferSync<Rank, 'string'>(sparseValues);
      const $defaultValue = util.decodeString(
          backend.data.get(defaultValue.dataId).values[0] as Uint8Array);
      outBuf = scatterImpl(
          indicesBuf, updatesBuf, outputShape, outputSize, sliceSize,
          numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
      break;
    }
    default:
      throw new Error(`Unsupported type ${sparseValues.dtype}`);
  }
  return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'cpu',
  kernelFunc: sparseToDense as {} as KernelFunc
};
