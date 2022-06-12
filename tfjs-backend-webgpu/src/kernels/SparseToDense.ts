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
import {ScatterOptimizedProgram} from '../scatter_optimized_webgpu';

import {identity} from './Identity';
import {reshape} from './Reshape';
import {tile} from './Tile';
import {zerosLike} from './ZerosLike';

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

  // If |sparseValues| is a scalar tensor, broadcast to the appropriate shape.
  const inferredSparseValuesShape = sparseIndices.shape.length > 1 ?
      sparseIndices.shape.slice(0, -1).concat(outputShape.slice(sliceRank)) :
      sparseIndices.shape.slice();
  const dimensionalSparseValues =
      sparseValues.shape.length === inferredSparseValuesShape.length ?
      identity({inputs: {x: sparseValues}, backend}) :
      reshape({
        inputs: {x: sparseValues},
        backend,
        attrs: {shape: new Array(inferredSparseValuesShape.length).fill(1)}
      });
  const $sparseValues =
      util.arraysEqual(
          inferredSparseValuesShape, dimensionalSparseValues.shape) ?
      identity({inputs: {x: dimensionalSparseValues}, backend}) :
      tile({
        inputs: {x: dimensionalSparseValues},
        backend,
        attrs: {reps: inferredSparseValuesShape}
      });
  backend.disposeData(dimensionalSparseValues.dataId);

  const flattenShape = [outputSize / sliceSize, sliceSize];

  const flattenIndices = reshape({
    inputs: {x: sparseIndices},
    backend,
    attrs: {shape: [numUpdates, sliceRank]}
  });
  const flattenValues = reshape({
    inputs: {x: $sparseValues},
    backend,
    attrs: {shape: [numUpdates, sliceSize]}
  });

  const type = flattenValues.dtype;

  // Fill output tensor with the default value.
  const dimensionalDefaultValue = reshape({
    inputs: {x: defaultValue},
    backend,
    attrs: {shape: new Array(flattenShape.length).fill(1)}
  });
  const flattenOutput = tile({
    inputs: {x: dimensionalDefaultValue},
    backend,
    attrs: {reps: flattenShape}
  });
  backend.disposeData(dimensionalDefaultValue.dataId);

  const size = util.sizeFromShape(flattenValues.shape);
  const uniformData = [
    {type: 'int32', data: [sliceRank]},
    {type: 'int32', data: strides},
    {type: 'int32', data: [size]},
  ];
  const program = new ScatterOptimizedProgram(
      flattenValues.shape, sliceRank, flattenIndices.shape.length,
      flattenValues.shape.length, strides, flattenShape, type);

  // First replace the default value with 0 at indices.
  const zeros = zerosLike({inputs: {x: flattenValues}, backend});
  backend.runWebGPUProgram(
      program, [zeros, flattenIndices, defaultValue], type, uniformData,
      flattenOutput);
  backend.disposeData(zeros.dataId);

  // Then replace 0 with the (sum of) sparse value(s) at indices.
  const zero =
      backend.makeTensorInfo([], type, util.makeZerosTypedArray(1, type));
  backend.runWebGPUProgram(
      program, [flattenValues, flattenIndices, zero], type, uniformData,
      flattenOutput);
  backend.disposeData(zero.dataId);

  const output = reshape(
      {inputs: {x: flattenOutput}, backend, attrs: {shape: outputShape}});

  backend.disposeData($sparseValues.dataId);
  backend.disposeData(flattenIndices.dataId);
  backend.disposeData(flattenValues.dataId);
  backend.disposeData(flattenOutput.dataId);
  return output;
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'webgpu',
  kernelFunc: sparseToDense as {} as KernelFunc
};
