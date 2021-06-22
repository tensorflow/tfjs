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
import {NumericDataType, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function grayscaleToRGB(args: {
  inputs: GrayscaleToRGBInputs,
  backend: MathBackendCPU
}): TensorInfo {
  const {inputs, backend} = args;
  const {image} = inputs;

  const [batch, height, width,] = image.shape;
  const outputChannel = 3;

  const outputShape = [batch, height, width, outputChannel];
  const output = util.getTypedArrayFromDType(
    image.dtype as NumericDataType, util.sizeFromShape(outputShape)
  );

  const imageValue = backend.data.get(image.dataId).values as TypedArray;

  let pixelIdx = 0;
  let outputIdx = 0;

  for (let batchIdx = 0; batchIdx < batch; batchIdx++) {

    for (let rowIdx = 0; rowIdx < height; rowIdx++) {

      for (let colIdx = 0; colIdx < width; colIdx++) {
        const pixel = imageValue[pixelIdx];

        for (let depthIdx = 0; depthIdx < 3; depthIdx++) {
          output[outputIdx] = pixel;
          outputIdx++;
        }

        pixelIdx++;
      }
    }
  }

  const dataId = backend.write(output, outputShape, image.dtype);
  return {dataId, shape: outputShape, dtype: image.dtype};
}

export const grayscaleToRGBConfig: KernelConfig = {
  kernelName: GrayscaleToRGB,
  backendName: 'cpu',
  kernelFunc: grayscaleToRGB as {} as KernelFunc
};
