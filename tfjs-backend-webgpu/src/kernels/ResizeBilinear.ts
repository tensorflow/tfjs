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

import {KernelConfig, KernelFunc, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ResizeBilinearProgram} from '../resize_bilinear_webgpu';

export function resizeBilinear(args: {
  inputs: ResizeBilinearInputs,
  backend: WebGPUBackend,
  attrs: ResizeBilinearAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, size, halfPixelCenters} = attrs;

  const [newHeight, newWidth] = size;
  const adjustHeight = alignCorners && newHeight > 1 ? 1.0 : 0.0;
  const adjustWidth = alignCorners && newWidth > 1 ? 1.0 : 0.0;
  const halfPixelCentersValue = halfPixelCenters ? 0.5 : 0.0;
  const uniformData = [
    {type: 'float32', data: [adjustHeight, adjustWidth]},
    {type: 'float32', data: [halfPixelCentersValue]}
  ];

  const program = new ResizeBilinearProgram(
      images.shape as [number, number, number, number], newHeight, newWidth);

  return backend.runWebGPUProgram(program, [images], 'float32', uniformData);
}

export const resizeBilinearConfig: KernelConfig = {
  kernelName: ResizeBilinear,
  backendName: 'webgpu',
  kernelFunc: resizeBilinear as {} as KernelFunc
};
