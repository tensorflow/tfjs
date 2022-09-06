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

import {KernelConfig, KernelFunc, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {UnaryOpType} from '../unary_op_util';
import {UnaryOpProgram} from '../unary_op_webgpu';

export function leakyRelu(args: {
  inputs: LeakyReluInputs,
  backend: WebGPUBackend,
  attrs: LeakyReluAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {alpha} = attrs;
  const uniformData = [{type: 'float32', data: [alpha]}];
  const program = new UnaryOpProgram(x.shape, UnaryOpType.LEAKYRELU);
  program.uniforms = 'alpha : f32,';
  return backend.runWebGPUProgram(program, [x], 'float32', uniformData);
}

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'webgpu',
  kernelFunc: leakyRelu as {} as KernelFunc
};
