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

import {KernelConfig, NumericDataType, TypedArray} from '@tensorflow/tfjs-core';
import {FlipLeftRight, FlipLeftRightInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const flipLeftRightConfig: KernelConfig = {
  kernelName: FlipLeftRight,
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {image} = inputs as FlipLeftRightInputs;
    const cpuBackend = backend as MathBackendCPU;

    const output = util.getTypedArrayFromDType(
        image.dtype as NumericDataType, util.sizeFromShape(image.shape));
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;

    const imageVals = cpuBackend.data.get(image.dataId).values as TypedArray;

    for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
      const batchOffset = batchIdx * imageWidth * imageHeight * numChannels;

      for (let row = 0; row < imageHeight; row++) {
        const rowOffset = row * (imageWidth * numChannels);

        for (let col = 0; col < imageWidth; col++) {
          const colOffset = col * numChannels;

          for (let channel = 0; channel < numChannels; channel++) {
            const coordX = Math.round(imageWidth - col - 1);
            const outIdx = batchOffset + rowOffset + colOffset + channel;

            let outputValue = imageVals[outIdx];
            // If the coordinate position falls within the image boundaries...
            if (coordX >= 0 && coordX < imageWidth) {
              // set the output to the image value at the coordinate position.
              const rotatedColOffset = coordX * numChannels;
              const imageIdx =
                  batchOffset + rowOffset + rotatedColOffset + channel;
              outputValue = imageVals[imageIdx];
            }
            output[outIdx] = outputValue;
          }
        }
      }
    }

    const dataId = cpuBackend.write(output, image.shape, image.dtype);
    return {dataId, shape: image.shape, dtype: image.dtype};
  }
};
