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

import {backend_util, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {GatherProgram} from '../gather_gpu';
import {gatherV2ImplCPU} from '../kernel_utils/shared';

import {reshape} from './Reshape';

export function gatherV2(args: {
  inputs: GatherV2Inputs,
  backend: MathBackendWebGL,
  attrs: GatherV2Attrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, indices} = inputs;
  const {axis, batchDims} = attrs;

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

  if (backend.shouldExecuteOnCPU([x, indices]) || x.dtype === 'string') {
    const indicesBuf = backend.bufferSync(flattenIndex);
    const xBuf = backend.bufferSync(flattenX);
    const outBuf = gatherV2ImplCPU(xBuf, indicesBuf, flattenOutputShape);

    toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));

    return backend.makeTensorInfo(
        shapeInfo.outputShape, outBuf.dtype, outBuf.values as TypedArray);
  }

  const program = new GatherProgram(flattenX.shape, flattenOutputShape);
  const res = backend.runWebGLProgram(
      program, [flattenX, flattenIndex], flattenX.dtype);
  toDispose.push(res);

  const reshaped = reshape(
      {inputs: {x: res}, backend, attrs: {shape: shapeInfo.outputShape}});
  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));
  return reshaped;
}

export const gatherV2Config: KernelConfig = {
  kernelName: GatherV2,
  backendName: 'webgl',
  kernelFunc: gatherV2 as {} as KernelFunc
};
