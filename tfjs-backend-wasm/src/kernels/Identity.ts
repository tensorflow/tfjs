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

import {Identity, IdentityInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

export function identity(args: {inputs: IdentityInputs, backend: BackendWasm}):
    TensorInfo {
  const {inputs: {x}, backend} = args;
  const out = backend.makeOutput(x.shape, x.dtype);
  const inVals = backend.typedArrayFromHeap(x);
  const outVals = backend.typedArrayFromHeap(out);
  outVals.set(inVals);
  return out;
}

export const identityConfig: KernelConfig = {
  kernelName: Identity,
  backendName: 'wasm',
  kernelFunc: identity as {} as KernelFunc,
};
