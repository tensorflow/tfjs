/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, LinSpace, LinSpaceAttrs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmLinSpace: (outId: number, start: number, stop: number, num: number) =>
    void;

function setup(backend: BackendWasm) {
  wasmLinSpace = backend.wasm.cwrap(LinSpace, null, [
    'number',  // outId
    'number',  // start
    'number',  // stop
    'number',  // num
  ]);
}

export function linSpace(args: {attrs: LinSpaceAttrs, backend: BackendWasm}):
    TensorInfo {
  const {attrs, backend} = args;
  const {start, stop, num} = attrs;

  // TFJS Cpu backend supports num as a float and returns undetermined tensor in
  // that case. However, according to TensorFlow spec, num should be a integer.
  const numInt = Math.floor(num);

  const out = backend.makeOutput([numInt], 'float32');
  wasmLinSpace(backend.dataIdMap.get(out.dataId).id, start, stop, numInt);
  return out;
}

export const linSpaceConfig: KernelConfig = {
  kernelName: LinSpace,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: linSpace as unknown as KernelFunc,
};
