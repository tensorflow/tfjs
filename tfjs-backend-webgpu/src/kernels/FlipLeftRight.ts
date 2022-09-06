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

import {KernelConfig, Tensor4D} from '@tensorflow/tfjs-core';
import {FlipLeftRight, FlipLeftRightInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {FlipLeftRightProgram} from '../flip_left_right_webgpu';

export const flipLeftRightConfig: KernelConfig = {
    kernelName: FlipLeftRight,
    backendName: 'webgpu',
    kernelFunc: ({inputs, backend}) => {
      const {image} = inputs as FlipLeftRightInputs;
      const webgpuBackend = backend as WebGPUBackend;

      const program = new FlipLeftRightProgram((image as Tensor4D).shape);
      const output =
          webgpuBackend.runWebGPUProgram(program, [image], image.dtype);
      return output;
  }
};
