/**
 * @license
 * Copyright 2022 Google LLC.
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

import {KernelConfig, KernelFunc, Step, StepAttrs, TensorInfo, UnaryInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {UnaryOpType} from '../unary_op_util';
import {UnaryOpProgram} from '../unary_op_webgpu';

export function step(
    {inputs, attrs, backend}:
        {inputs: UnaryInputs, attrs: StepAttrs, backend: WebGPUBackend}):
    TensorInfo {
  const {x} = inputs;
  const program =
      new UnaryOpProgram(x.shape, UnaryOpType.STEP, 'stepAlpha : f32,');
  const uniformData = [{type: 'float32', data: [attrs.alpha]}];
  return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}

export const stepConfig: KernelConfig = {
  kernelName: Step,
  backendName: 'webgpu',
  kernelFunc: step as unknown as KernelFunc
};
