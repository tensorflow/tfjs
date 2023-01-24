/**
 * @license
 * Copyright 2023 Google LLC.
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
import {backend_util, BroadcastArgs, BroadcastArgsInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

export function broadcastArgs(args: {
  inputs: BroadcastArgsInputs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend} = args;
  const {s0, s1} = inputs;

  const s0Vals = backend.typedArrayFromHeap(s0);
  const s1Vals = backend.typedArrayFromHeap(s1);

  const broadcastShape = backend_util.assertAndGetBroadcastShape(
      Array.from(s0Vals), Array.from(s1Vals));

  return backend.makeOutput(
      [broadcastShape.length], 'int32', /*memoryOffset=*/undefined,
      /*values=*/new Int32Array(broadcastShape));
}

export const broadcastArgsConfig: KernelConfig = {
  kernelName: BroadcastArgs,
  backendName: 'wasm',
  kernelFunc: broadcastArgs
};
