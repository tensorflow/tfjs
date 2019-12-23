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
    (xId: number, dtype: CppDType, xStrides: Uint8Array, stridesSize: number,
     indicesId: number, axis: number, outStrides: Uint8Array, outId: number) =>
        void;

function setup(backend: BackendWasm): void {
  wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
    'number',  // xId
    'number',  // dtype
    'array',   // xStrides
    'number',  // stridesSize
    'number',  // indicesId
    'number',  // axis
    'array',   // outStrides
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
  const stridesSize = x.shape.length - 1;

  const out = backend.makeOutput(newShape, x.dtype);
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const xData = backend.dataIdMap.get(x.dataId);
  const xId = xData.id;

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const outId = backend.dataIdMap.get(out.dataId).id;

  const xStridesBytes =
      new Uint8Array(new Int32Array(util.computeStrides(x.shape)).buffer);
  const outStridesBytes =
      new Uint8Array(new Int32Array(util.computeStrides(newShape)).buffer);

  wasmGather(
      xId, CppDType[x.dtype], xStridesBytes, stridesSize, indicesId, axis,
      outStridesBytes, outId);

  return out;
}

registerKernel({
  kernelName: 'Gather',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: gather
});
