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

import {KernelConfig, KernelFunc, ResizeNearestNeighbor, ResizeNearestNeighborAttrs, ResizeNearestNeighborInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ResizeNearestNeighborProgram} from '../resize_nearest_neighbor_webgpu';

export function resizeNearestNeighbor(args: {
  inputs: ResizeNearestNeighborInputs,
  backend: WebGPUBackend,
  attrs: ResizeNearestNeighborAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, halfPixelCenters, size} = attrs;

  const [newHeight, newWidth] = size;
  const adjustHeight = alignCorners && newHeight > 1 ? 1.0 : 0.0;
  const adjustWidth = alignCorners && newWidth > 1 ? 1.0 : 0.0;
  // When align corners is false, we rounds the value with floor.
  const roundBase = alignCorners ? 0.5 : 0.0;
  const uniformData = [
    {type: 'float32', data: [adjustHeight, adjustWidth]},
    {type: 'float32', data: [roundBase]}
  ];

  const program = new ResizeNearestNeighborProgram(
      images.shape as [number, number, number, number], newHeight, newWidth,
      halfPixelCenters);
  return backend.runWebGPUProgram(program, [images], images.dtype, uniformData);
}

export const resizeNearestNeighborConfig: KernelConfig = {
  kernelName: ResizeNearestNeighbor,
  backendName: 'webgpu',
  kernelFunc: resizeNearestNeighbor as {} as KernelFunc
};
