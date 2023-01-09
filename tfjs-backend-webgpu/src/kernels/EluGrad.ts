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

import {EluGrad, EluGradInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {BinaryOpType} from '../binary_op_util';
import {BinaryOpProgram} from '../binary_op_webgpu';

export const eluGrad =
    (args: {inputs: EluGradInputs, backend: WebGPUBackend}): TensorInfo => {
      const {inputs, backend} = args;
      const {dy, y} = inputs;

      const program =
          new BinaryOpProgram(BinaryOpType.ELU_DER, dy.shape, y.shape);
      return backend.runWebGPUProgram(program, [dy, y], dy.dtype);
    };

export const eluGradConfig: KernelConfig = {
  kernelName: EluGrad,
  backendName: 'webgpu',
  kernelFunc: eluGrad as unknown as KernelFunc
};
