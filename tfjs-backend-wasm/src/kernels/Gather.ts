/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {CppDType} from './types';

interface GatherInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  indices: TensorInfo;
}

interface GatherAttrs extends NamedAttrMap {
  axis: number;
}

let wasmGather:
    (xId: number, dtype: CppDType, xShape: Uint8Array, rank: number,
     indicesId: number, axis: number, outShape: Uint8Array, outId: number) =>
        void;

function setup(backend: BackendWasm): void {
  wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
    'number',  // xId
    'number',  // dtype
    'array',   // x.shape
    'number',  // rank
    'number',  // indicesId
    'number',  // axis
    'array',   // out.shape
    'number'   // outId
  ]);
}

function gather(
    args: {backend: BackendWasm, inputs: GatherInputs, attrs: GatherAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {x, indices} = inputs;
  const {axis} = attrs;

  const newShape = x.shape.slice();
  newShape[axis] = util.sizeFromShape(indices.shape);
  const rank = x.shape.length;

  const out = backend.makeOutput(newShape, x.dtype);
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const xData = backend.dataIdMap.get(x.dataId);
  const xId = xData.id;

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const outId = backend.dataIdMap.get(out.dataId).id;

  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
  const outShapeBytes = new Uint8Array(new Int32Array(newShape).buffer);
  wasmGather(
      xId, CppDType[x.dtype], xShapeBytes, rank, indicesId, axis, outShapeBytes,
      outId);

  return out;
}

registerKernel({
  kernelName: 'Gather',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: gather
});
