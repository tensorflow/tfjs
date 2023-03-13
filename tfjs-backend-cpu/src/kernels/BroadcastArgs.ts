/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, BroadcastArgs, BroadcastArgsInputs, KernelConfig, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function broadcastArgs(args: {
  inputs: BroadcastArgsInputs,
  backend: MathBackendCPU,
}): TensorInfo {
  const {inputs, backend} = args;
  const {s0, s1} = inputs;

  const s0Vals = backend.data.get(s0.dataId).values as TypedArray;
  const s1Vals = backend.data.get(s1.dataId).values as TypedArray;

  const broadcastShape = backend_util.assertAndGetBroadcastShape(
      Array.from(s0Vals), Array.from(s1Vals));

  return backend.makeTensorInfo(
      [broadcastShape.length], 'int32', Int32Array.from(broadcastShape));
}

export const broadcastArgsConfig: KernelConfig = {
  kernelName: BroadcastArgs,
  backendName: 'cpu',
  kernelFunc: broadcastArgs
};
