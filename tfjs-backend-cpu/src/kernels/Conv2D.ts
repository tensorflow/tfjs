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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function conv2D(
    args: {inputs: Conv2DInputs, backend: MathBackendCPU, attrs: Conv2DAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dataFormat, dilations, dimRoundingMode} = attrs;

  assertNotComplex([x, filter], 'conv2d');

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);

  const filterHeight = convInfo.filterHeight;
  const filterWidth = convInfo.filterWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const padLeft = convInfo.padInfo.left;
  const padTop = convInfo.padInfo.top;
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';

  const y = new TensorBuffer(convInfo.outShape, x.dtype as 'float32');

  const xStrides = util.computeStrides(x.shape);
  const filterStrides = util.computeStrides(filter.shape);

  const xBatchStride = xStrides[0];
  const xRowStride = isChannelsLast ? xStrides[1] : xStrides[2];
  const xColStride = isChannelsLast ? xStrides[2] : 1;
  const xChannelStride = isChannelsLast ? 1 : xStrides[1];
  const yBatchStride = y.strides[0];
  const yRowStride = isChannelsLast ? y.strides[1] : y.strides[2];
  const yColStride = isChannelsLast ? y.strides[2] : 1;
  const yChannelStride = isChannelsLast ? 1 : y.strides[1];

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const wVals = backend.data.get(filter.dataId).values as TypedArray;
  const yVals = y.values;

  for (let b = 0; b < convInfo.batchSize; ++b) {
    const xOffset1 = b * xBatchStride;
    const yOffset1 = b * yBatchStride;
    for (let yR = 0; yR < convInfo.outHeight; ++yR) {
      const yOffset2 = yOffset1 + yR * yRowStride;
      const xRCorner = yR * convInfo.strideHeight - padTop;
      for (let wR = 0; wR < filterHeight; ++wR) {
        const xR = xRCorner + wR * dilationHeight;
        if (xR < 0 || xR >= convInfo.inHeight) {
          continue;
        }
        const wOffset1 = wR * filterStrides[0];
        const xOffset2 = xOffset1 + xR * xRowStride;
        for (let yC = 0; yC < convInfo.outWidth; ++yC) {
          const yOffset3 = yOffset2 + yC * yColStride;
          const xCCorner = yC * convInfo.strideWidth - padLeft;
          for (let wC = 0; wC < filterWidth; ++wC) {
            const xC = xCCorner + wC * dilationWidth;
            if (xC < 0 || xC >= convInfo.inWidth) {
              continue;
            }
            const wOffset2 = wOffset1 + wC * filterStrides[1];
            const xOffset3 = xOffset2 + xC * xColStride;
            let wOffset3 = wOffset2;
            for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
              const xVal = xVals[xOffset3 + d1 * xChannelStride];
              for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                yVals[yOffset3 + d2 * yChannelStride] +=
                    xVal * wVals[wOffset3 + d2];
              }
              wOffset3 += convInfo.outChannels;
            }
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(y.shape, y.dtype, yVals);
}

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'cpu',
  kernelFunc: conv2D as {} as KernelFunc
};
