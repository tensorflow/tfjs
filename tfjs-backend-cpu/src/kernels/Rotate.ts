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
import {Rotate, RotateAttrs, RotateInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const rotateConfig: KernelConfig = {
  kernelName: Rotate,
  backendName: 'cpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {image} = inputs as RotateInputs;
    const {radians, fillValue, center} = attrs as {} as RotateAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const output = util.getTypedArrayFromDType(
        image.dtype as NumericDataType, util.sizeFromShape(image.shape));
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;

    const centerX =
        imageWidth * (typeof center === 'number' ? center : center[0]);
    const centerY =
        imageHeight * (typeof center === 'number' ? center : center[1]);

    const sinFactor = Math.sin(-radians);
    const cosFactor = Math.cos(-radians);
    const imageVals = cpuBackend.data.get(image.dataId).values as TypedArray;

    for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
      for (let row = 0; row < imageHeight; row++) {
        for (let col = 0; col < imageWidth; col++) {
          for (let channel = 0; channel < numChannels; channel++) {
            const coords = [batch, row, col, channel];

            const x = coords[2];
            const y = coords[1];

            let coordX = (x - centerX) * cosFactor - (y - centerY) * sinFactor;
            let coordY = (x - centerX) * sinFactor + (y - centerY) * cosFactor;

            coordX = Math.round(coordX + centerX);
            coordY = Math.round(coordY + centerY);

            let outputValue = fillValue;
            if (typeof fillValue !== 'number') {
              if (channel === 3) {
                outputValue = 255;
              } else {
                outputValue = fillValue[channel];
              }
            }

            if (coordX >= 0 && coordX < imageWidth && coordY >= 0 &&
                coordY < imageHeight) {
              const imageIdx =
                  batchIdx * imageWidth * imageHeight * numChannels +
                  coordY * (imageWidth * numChannels) + coordX * numChannels +
                  channel;
              outputValue = imageVals[imageIdx];
            }

            const outIdx = batchIdx * imageWidth * imageHeight * numChannels +
                row * (imageWidth * numChannels) + col * numChannels + channel;
            output[outIdx] = outputValue as number;
          }
        }
      }
    }

    const dataId = cpuBackend.write(output, image.shape, image.dtype);
    return {dataId, shape: image.shape, dtype: image.dtype};
  }
};
