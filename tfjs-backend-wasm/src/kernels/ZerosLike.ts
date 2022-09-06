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

import {KernelConfig, KernelFunc, ZerosLike, ZerosLikeInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

function zerosLike(args: {inputs: ZerosLikeInputs, backend: BackendWasm}) {
  const {inputs: {x}, backend} = args;
  const out = backend.makeOutput(x.shape, x.dtype);
  const outVals = backend.typedArrayFromHeap(out);
  outVals.fill(0);
  return out;
}

export const zerosLikeConfig: KernelConfig = {
  kernelName: ZerosLike,
  backendName: 'wasm',
  kernelFunc: zerosLike as {} as KernelFunc,
};
