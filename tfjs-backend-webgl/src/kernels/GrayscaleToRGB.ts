
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

import {KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';
import {GrayscaleToRGB, GrayscaleToRGBInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {GrayscaleToRGBProgram} from '../grayscale_to_rgb_gpu';

export function grayscaleToRGB(args: {
  inputs: GrayscaleToRGBInputs,
  backend: MathBackendWebGL
}): TensorInfo {
  const {inputs, backend} = args;
  const {image} = inputs;

  const program = new GrayscaleToRGBProgram(
    image.shape as [number, number, number, number]
  );
  const output = backend.runWebGLProgram(program, [image], image.dtype);

  return output;
}

export const grayscaleToRGBConfig: KernelConfig = {
  kernelName: GrayscaleToRGB,
  backendName: 'webgl',
  kernelFunc: grayscaleToRGB as {} as KernelFunc
};
