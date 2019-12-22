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

import {NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface GatherInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

let wasmGather: (xId: number, indicesId: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmGather = backend.wasm.cwrap('Gather', null /*void*/, [
    'number',  // xId
    'number',  // indicesId
    'number'   // outId
  ]);
}

function gather(args: {backend: BackendWasm, inputs: GatherInputs}):
    TensorInfo {
  const {backend, inputs} = args;
  const {x, indices} = inputs;

  const out = backend.makeOutput([], x.dtype);
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const xData = backend.dataIdMap.get(x.dataId);
  const xId = xData.id;

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmGather(xId, indicesId, outId);

  return out;
}

registerKernel({
  kernelName: 'Gather',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: gather
});
