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

import {backend_util, buffer, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, KernelFunc, Rank, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {gatherV2ImplCPU} from '../kernel_utils/shared';

import {GatherProgram} from '../gather_webgpu';
import {reshape} from './Reshape';

export function gatherV2(
    args:
        {inputs: GatherV2Inputs, backend: WebGPUBackend, attrs: GatherV2Attrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, indices} = inputs;
  const {axis, batchDims} = attrs;

  // Unlike WebGL, WebGPU won't check if index is out of bound by calling
  // backend.readSync() function in debug mode.
  const parsedAxis = util.parseAxisParam(axis, x.shape)[0];

  const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(
      x, indices, parsedAxis, batchDims);

  const indicesSize = util.sizeFromShape(indices.shape);

  const toDispose = [];

  const flattenX = reshape({
    inputs: {x},
    backend,
    attrs: {
      shape: [
        shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
        shapeInfo.sliceSize
      ]
    }
  });

  const flattenIndex = reshape({
    inputs: {x: indices},
    backend,
    attrs: {shape: [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize]}
  });

  toDispose.push(flattenX);
  toDispose.push(flattenIndex);

  const flattenOutputShape = [
    shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
    shapeInfo.sliceSize
  ];

  if (backend.shouldExecuteOnCPU([x, indices])) {
    const indicesBufferInfo = backend.tensorMap.get(flattenIndex.dataId);
    const indicesValues = indicesBufferInfo.values as TypedArray;
    const indicesBuf =
        buffer(flattenIndex.shape, flattenIndex.dtype, indicesValues) as
        TensorBuffer<Rank>;
    const xBufferInfo = backend.tensorMap.get(flattenX.dataId);
    const xValues = xBufferInfo.values as TypedArray;
    const xBuf =
        buffer(flattenX.shape, flattenX.dtype, xValues) as TensorBuffer<Rank>;
    const outBuf = gatherV2ImplCPU(xBuf, indicesBuf, flattenOutputShape);

    toDispose.forEach(t => backend.disposeData(t.dataId));

    return backend.makeTensorInfo(
        shapeInfo.outputShape, outBuf.dtype, outBuf.values as TypedArray);
  }

  const program = new GatherProgram(flattenX.shape, flattenOutputShape);
  const res = backend.runWebGPUProgram(
      program, [flattenX, flattenIndex], flattenX.dtype);
  toDispose.push(res);

  const reshaped = reshape(
      {inputs: {x: res}, backend, attrs: {shape: shapeInfo.outputShape}});
  toDispose.forEach(t => backend.disposeData(t.dataId));
  return reshaped;
}

export const gatherV2Config: KernelConfig = {
  kernelName: GatherV2,
  backendName: 'webgpu',
  kernelFunc: gatherV2 as {} as KernelFunc
};
