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
import {backend_util, RotateWithOffset, RotateWithOffsetAttrs, RotateWithOffsetInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const rotateWithOffsetConfig: KernelConfig = {
  kernelName: RotateWithOffset,
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {image} = inputs as RotateWithOffsetInputs;
    const {radians, fillValue, center} = attrs as {} as RotateWithOffsetAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const output = util.getTypedArrayFromDType(
        image.dtype as NumericDataType, util.sizeFromShape(image.shape));
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;

    const [centerX, centerY] =
        backend_util.getImageCenter(center, imageHeight, imageWidth);
    const fullOpacityValue = 255;

    const sinFactor = Math.sin(radians);
    const cosFactor = Math.cos(radians);
    const imageVals = cpuBackend.data.get(image.dataId).values as TypedArray;

    for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
      const batchOffset = batchIdx * imageWidth * imageHeight * numChannels;

      for (let row = 0; row < imageHeight; row++) {
        const rowOffset = row * (imageWidth * numChannels);

        for (let col = 0; col < imageWidth; col++) {
          const colOffset = col * numChannels;

          for (let channel = 0; channel < numChannels; channel++) {
            const coords = [batch, row, col, channel];

            const x = coords[2];
            const y = coords[1];

            // coordX/coordY are the result of rotating and translating x/y.
            let coordX = (x - centerX) * cosFactor - (y - centerY) * sinFactor;
            let coordY = (x - centerX) * sinFactor + (y - centerY) * cosFactor;
            coordX = Math.round(coordX + centerX);
            coordY = Math.round(coordY + centerY);

            let outputValue = fillValue;
            if (typeof fillValue !== 'number') {
              if (channel === 3) {
                outputValue = fullOpacityValue;
              } else {
                outputValue = fillValue[channel];
              }
            }

            // If the coordinate position falls within the image boundaries...
            if (coordX >= 0 && coordX < imageWidth && coordY >= 0 &&
                coordY < imageHeight) {
              // set the output to the image value at the coordinate position.
              const rotatedRowOffset = coordY * (imageWidth * numChannels);
              const rotatedColOffset = coordX * numChannels;
              const imageIdx =
                  batchOffset + rotatedRowOffset + rotatedColOffset + channel;
              outputValue = imageVals[imageIdx];
            }

            const outIdx = batchOffset + rowOffset + colOffset + channel;
            output[outIdx] = outputValue as number;
          }
        }
      }
    }

    const dataId = cpuBackend.write(output, image.shape, image.dtype);
    return {dataId, shape: image.shape, dtype: image.dtype};
  }
};
