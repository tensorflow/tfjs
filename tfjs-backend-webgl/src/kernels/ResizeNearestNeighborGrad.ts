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

import {KernelConfig, KernelFunc, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ResizeNearestNeigborBackpropProgram} from '../resize_nearest_neighbor_backprop_gpu';

export function resizeNearestNeighborGrad(args: {
  inputs: ResizeNearestNeighborGradInputs,
  backend: MathBackendWebGL,
  attrs: ResizeNearestNeighborGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images, dy} = inputs;
  const {alignCorners} = attrs;

  const program = new ResizeNearestNeigborBackpropProgram(
      dy.shape as [number, number, number, number],
      images.shape as [number, number, number, number], alignCorners);
  return backend.runWebGLProgram(program, [dy], dy.dtype);
}

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'webgl',
  kernelFunc: resizeNearestNeighborGrad as {} as KernelFunc
};
