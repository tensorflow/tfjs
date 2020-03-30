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

import {KernelConfig, SquaredDifference, SquaredDifferenceInputs, Tensor} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {BinaryOpProgram, SQUARED_DIFFERENCE} from './binary_op_webgpu';

export const squaredDifferenceConfig: KernelConfig = {
  kernelName: SquaredDifference,
  backendName: 'webgpu',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs as SquaredDifferenceInputs;
    const webGPUBackend = backend as WebGPUBackend;

    const program = new BinaryOpProgram(SQUARED_DIFFERENCE, a.shape, b.shape);
    return webGPUBackend.compileAndRun(program, [a as Tensor, b as Tensor]);
  }
};
