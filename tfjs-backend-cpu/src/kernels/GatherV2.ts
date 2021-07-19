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

import {backend_util, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {gatherV2Impl} from './GatherV2_impl';
import {reshape} from './Reshape';

export function gatherV2(args: {
  inputs: GatherV2Inputs,
  backend: MathBackendCPU,
  attrs: GatherV2Attrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, indices} = inputs;
  const {axis, batchDims} = attrs;

  assertNotComplex([x, indices], 'gatherV2');

  let $batchDims = batchDims;

  if (batchDims == null) {
    $batchDims = 0;
  }

  const indicesSize = util.sizeFromShape(indices.shape);

  const parsedAxis = util.parseAxisParam(axis, x.shape)[0];
  const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(
      x, indices, parsedAxis, $batchDims);

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

  const flattenOutputShape = [
    shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
    shapeInfo.sliceSize
  ];

  const indicesBuf = backend.bufferSync(flattenIndex);
  const xBuf = backend.bufferSync(flattenX);
  const outBuf = gatherV2Impl(xBuf, indicesBuf, flattenOutputShape);

  backend.disposeIntermediateTensorInfo(flattenX);
  backend.disposeIntermediateTensorInfo(flattenIndex);

  return backend.makeTensorInfo(
      shapeInfo.outputShape, outBuf.dtype, outBuf.values);
}

export const gatherV2Config: KernelConfig = {
  kernelName: GatherV2,
  backendName: 'cpu',
  kernelFunc: gatherV2 as {} as KernelFunc
};
