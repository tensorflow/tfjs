/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {backend_util, GatherV2, GatherV2Attrs, GatherV2Inputs, KernelConfig, KernelFunc, Tensor, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {reshape} from './Reshape';
import {CppDType} from './types';

let wasmGather: (
    xId: number, dtype: CppDType, xStrides: Uint8Array, stridesSize: number,
    indicesId: number, batchSize: number, outStrides: Uint8Array,
    outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
    'number',  // xId
    'number',  // dtype
    'array',   // xStrides
    'number',  // stridesSize
    'number',  // indicesId
    'number',  // batchSize
    'array',   // outStrides
    'number'   // outId
  ]);
}

function gatherV2(
    args: {backend: BackendWasm, inputs: GatherV2Inputs, attrs: GatherV2Attrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {x, indices} = inputs;
  const {axis, batchDims} = attrs;

  // Throw error when any index is out of bound.
  const parsedAxis = util.parseAxisParam(axis, x.shape)[0];
  const indicesVals = backend.readSync(indices.dataId) as TypedArray;
  const axisDim = x.shape[parsedAxis];
  for (let i = 0; i < indicesVals.length; ++i) {
    const index = indicesVals[i];
    util.assert(
        index <= axisDim - 1 && index >= 0,
        () =>
            `GatherV2: the index value ${index} is not in [0, ${axisDim - 1}]`);
  }

  const shapeInfo = backend_util.segment_util.collectGatherOpShapeInfo(
      x as Tensor, indices as Tensor, parsedAxis, batchDims);

  const flattenX = reshape({
    inputs: {x},
    attrs: {
      shape: [
        shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
        shapeInfo.sliceSize
      ]
    },
    backend
  });
  const indicesSize = util.sizeFromShape(indices.shape);
  const flattenIndex = reshape({
    inputs: {x: indices},
    attrs: {shape: [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize]},
    backend
  });
  const flattenOutputShape = [
    shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
    shapeInfo.sliceSize
  ];

  const out = backend.makeOutput(flattenOutputShape, x.dtype);
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }
  const stridesSize = flattenX.shape.length - 1;

  const xData = backend.dataIdMap.get(flattenX.dataId);
  const xId = xData.id;

  const indicesData = backend.dataIdMap.get(flattenIndex.dataId);
  const indicesId = indicesData.id;

  const outId = backend.dataIdMap.get(out.dataId).id;

  const xStridesBytes = new Uint8Array(
      new Int32Array(util.computeStrides(flattenX.shape)).buffer);
  const outStridesBytes = new Uint8Array(
      new Int32Array(util.computeStrides(flattenOutputShape)).buffer);

  wasmGather(
      xId, CppDType[x.dtype], xStridesBytes, stridesSize, indicesId,
      shapeInfo.batchSize, outStridesBytes, outId);

  backend.disposeData(flattenX.dataId);
  backend.disposeData(flattenIndex.dataId);

  // reshape
  out.shape = shapeInfo.outputShape;
  return out;
}

export const gatherV2Config: KernelConfig = {
  kernelName: GatherV2,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: gatherV2 as {} as KernelFunc
};
