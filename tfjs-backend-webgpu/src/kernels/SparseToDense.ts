/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {WebGPUBackend} from '../backend_webgpu';
import {scatterImplCPU} from '../kernel_utils/shared';
import {ScatterProgram} from '../scatter_webgpu';

import {identity} from './Identity';
import {reshape} from './Reshape';
import {tile} from './Tile';

export function sparseToDense(args: {
  inputs: SparseToDenseInputs,
  backend: WebGPUBackend,
  attrs: SparseToDenseAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sparseIndices, sparseValues, defaultValue} = inputs;
  const {outputShape} = attrs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);

  const sumDupeIndices = false;
  if (sparseValues.dtype === 'string') {
    const indicesBuf = backend.bufferSync<Rank, 'int32'>(sparseIndices);
    const updatesBuf = backend.bufferSync<Rank, 'string'>(sparseValues);
    const $defaultValue = util.decodeString(
        backend.readSync(defaultValue.dataId)[0] as Uint8Array);
    const outBuf = scatterImplCPU(
        indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates,
        sliceRank, strides, $defaultValue, sumDupeIndices);
    return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
  }

  const flattenShape = [outputSize / sliceSize, sliceSize];

  const $sparseIndices = reshape({
    inputs: {x: sparseIndices},
    backend,
    attrs: {shape: [numUpdates, sliceRank]}
  });
  const $sparseValues = sparseValues.shape.length ?
      reshape({
        inputs: {x: sparseValues},
        backend,
        attrs: {shape: [numUpdates, sliceSize]}
      }) :
      identity({inputs: {x: sparseValues}, backend});

  const type = $sparseValues.dtype;
  const zero =
      backend.makeTensorInfo([], type, util.makeZerosTypedArray(1, type));

  // Fill output tensor with the default value.
  const $defaultValue = reshape({
    inputs: {x: defaultValue},
    backend,
    attrs: {shape: Array(flattenShape.length).fill(1)}
  });
  const $denseValues =
      tile({inputs: {x: $defaultValue}, backend, attrs: {reps: flattenShape}});

  const size = util.sizeFromShape([numUpdates, sliceSize]);
  const uniformData = [
    {type: 'int32', data: [sliceRank]},
    {type: 'int32', data: strides},
    {type: 'int32', data: [size]},
  ];

  switch (numUpdates) {
    case 0:
      break;
    case 1:
      if (true) {
        const program = new ScatterProgram(
            [numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length,
            $sparseValues.shape.length, strides, flattenShape, type,
            sumDupeIndices);
        backend.runWebGPUProgram(
            program, [$sparseValues, $sparseIndices], type, uniformData,
            $denseValues);
      }
      break;
    default:
      if (true) {
        // First replace the default value with 0 at indices.
        const program = new ScatterProgram(
            [numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length,
            zero.shape.length, strides, flattenShape, type, sumDupeIndices);
        backend.runWebGPUProgram(
            program, [zero, $sparseIndices], type, uniformData, $denseValues);
      }
      {
        // Then replace 0 with the (sum of) sparse value(s) at indices.
        const program = new ScatterProgram(
            [numUpdates, sliceSize], sliceRank, $sparseIndices.shape.length,
            $sparseValues.shape.length, strides, flattenShape, type);
        backend.runWebGPUProgram(
            program, [$sparseValues, $sparseIndices], type, uniformData,
            $denseValues);
      }
  }

  const denseValues = reshape(
      {inputs: {x: $denseValues}, backend, attrs: {shape: outputShape}});

  backend.disposeData($sparseIndices.dataId);
  backend.disposeData($sparseValues.dataId);
  backend.disposeData($defaultValue.dataId);
  backend.disposeData(zero.dataId);
  backend.disposeData($denseValues.dataId);
  return denseValues;
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'webgpu',
  kernelFunc: sparseToDense as unknown as KernelFunc
};
