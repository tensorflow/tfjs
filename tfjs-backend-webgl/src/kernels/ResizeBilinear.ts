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

import {env, KernelConfig, KernelFunc, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ResizeBilinearProgram} from '../resize_bilinear_gpu';
import {ResizeBilinearPackedProgram} from '../resize_bilinear_packed_gpu';

export function resizeBilinear(args: {
  inputs: ResizeBilinearInputs,
  backend: MathBackendWebGL,
  attrs: ResizeBilinearAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, halfPixelCenters, size} = attrs;

  const [newHeight, newWidth] = size;

  const program = env().getBool('WEBGL_PACK_IMAGE_OPERATIONS') ?
      new ResizeBilinearPackedProgram(
          images.shape as [number, number, number, number], newHeight, newWidth,
          alignCorners, halfPixelCenters) :
      new ResizeBilinearProgram(
          images.shape as [number, number, number, number], newHeight, newWidth,
          alignCorners, halfPixelCenters);
  return backend.runWebGLProgram(program, [images], 'float32');
}

export const resizeBilinearConfig: KernelConfig = {
  kernelName: ResizeBilinear,
  backendName: 'webgl',
  kernelFunc: resizeBilinear as {} as KernelFunc
};
