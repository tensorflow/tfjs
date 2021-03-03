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

import {KernelConfig, KernelFunc, Prelu, PreluInputs, TensorInfo} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {BinaryOpProgram} from './binary_op_webgpu';
import {BinaryOpType, getBinaryOpString} from './binary_ops';

export function prelu(args: {inputs: PreluInputs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x, alpha} = inputs;

  const program = new BinaryOpProgram(
      getBinaryOpString(BinaryOpType.PRELU), x.shape, alpha.shape);
  return backend.runWebGPUProgram(program, [x, alpha], x.dtype);
}

export const preluConfig: KernelConfig = {
  kernelName: Prelu,
  backendName: 'webgpu',
  kernelFunc: prelu as {} as KernelFunc
};
