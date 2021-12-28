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

import {backend_util, KernelConfig, Tensor4D} from '@tensorflow/tfjs-core';
import {RotateWithOffset, RotateWithOffsetAttrs, RotateWithOffsetInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {RotateProgram} from '../rotate_webgpu';

export const rotateWithOffsetConfig: KernelConfig = {
    kernelName: RotateWithOffset,
    backendName: 'webgpu',
    kernelFunc: ({inputs, attrs, backend}) => {
      const {image} = inputs as RotateWithOffsetInputs;
      const {radians, fillValue, center} = attrs as {} as RotateWithOffsetAttrs;
      const webgpuBackend = backend as WebGPUBackend;

      const program = new RotateProgram((image as Tensor4D).shape, fillValue);
      const [centerX, centerY] =
          backend_util.getImageCenter(center, image.shape[1], image.shape[2]);
      const uniformData = [
            {type: 'float32', data: [centerX]},
            {type: 'float32', data: [centerY]},
            {type: 'float32', data: [Math.sin(radians)]},
            {type: 'float32', data: [Math.cos(radians)]}
          ];

      if (typeof fillValue === 'number') {
        uniformData.push(
            {type: 'float32', data: [Number.parseFloat(fillValue.toFixed(2))]});
      } else {
        uniformData.push({type: 'float32', data: fillValue});
      }

      const output = webgpuBackend.runWebGPUProgram(
          program, [image], image.dtype, uniformData);
      return output;
   }
 };
