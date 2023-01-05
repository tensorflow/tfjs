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

import {BroadcastArgs, BroadcastArgsInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {BroadcastArgsProgram} from '../broadcast_args_webgpu';

export function broadcastArgs(args: {
  inputs: BroadcastArgsInputs,
  backend: WebGPUBackend,
}): TensorInfo {
  const {inputs, backend} = args;
  const {s0, s1} = inputs;

  const s0Size = util.sizeFromShape(s0.shape);
  const s1Size = util.sizeFromShape(s1.shape);
  const outputSize = Math.max(s0Size, s1Size);

  const program = new BroadcastArgsProgram(outputSize);
  const uniformData =
      [{type: 'int32', data: [s0Size]}, {type: 'int32', data: [s1Size]}];
  return backend.runWebGPUProgram(program, [s0, s1], 'int32', uniformData);
}

export const broadcastArgsConfig: KernelConfig = {
  kernelName: BroadcastArgs,
  backendName: 'webgpu',
  kernelFunc: broadcastArgs as unknown as KernelFunc
};
