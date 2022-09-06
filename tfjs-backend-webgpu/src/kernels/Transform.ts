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

import {KernelConfig, KernelFunc, TensorInfo, Transform, TransformAttrs, TransformInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {TransformProgram} from '../transform_webgpu';

export function transform(args: {
  inputs: TransformInputs,
  backend: WebGPUBackend,
  attrs: TransformAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image, transforms} = inputs;
  const {interpolation, fillMode, fillValue, outputShape} = attrs;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;
  const [outHeight, outWidth] =
      outputShape != null ? outputShape : [imageHeight, imageWidth];
  const outShape =
      [batch, outHeight, outWidth,
       numChannels] as [number, number, number, number];

  const program = new TransformProgram(outShape);
  const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
  let fillModeId: number;
  switch (fillMode) {
    case 'constant':
      fillModeId = 1;
      break;
    case 'reflect':
      fillModeId = 2;
      break;
    case 'wrap':
      fillModeId = 3;
      break;
    case 'nearest':
      fillModeId = 4;
      break;
    default:
      fillModeId = 1;
      break;
  }
  const uniformData = [
    {type: 'int32', data: [interpolationModeId]},
    {type: 'int32', data: [fillModeId]}, {type: 'float32', data: [fillValue]}
  ];
  return backend.runWebGPUProgram(
      program, [image, transforms], 'float32', uniformData);
}

export const transformConfig: KernelConfig = {
  kernelName: Transform,
  backendName: 'webgpu',
  kernelFunc: transform as {} as KernelFunc
};
