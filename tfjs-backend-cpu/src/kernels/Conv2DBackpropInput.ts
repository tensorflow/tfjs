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

import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function conv2DBackpropInput(args: {
  inputs: Conv2DBackpropInputInputs,
  backend: MathBackendCPU,
  attrs: Conv2DBackpropInputAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {inputShape, strides, pad, dataFormat, dimRoundingMode} = attrs;

  assertNotComplex([dy, filter], 'conv2dBackpropInput');

  const filterStrides = util.computeStrides(filter.shape);
  const dyStrides = util.computeStrides(dy.shape);

  let $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);

  const dx = new TensorBuffer(convInfo.inShape, 'float32');
  const dxValues = dx.values;
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  const fltValues = backend.data.get(filter.dataId).values as TypedArray;
  const [fltS0, fltS1, fltS2] = filterStrides;
  const {
    batchSize,
    filterHeight,
    filterWidth,
    inChannels,
    inHeight,
    inWidth,
    outChannels,
    outHeight,
    outWidth,
    strideHeight,
    strideWidth
  } = convInfo;
  $dataFormat = convInfo.dataFormat;
  const topPad = filterHeight - 1 - convInfo.padInfo.top;
  const leftPad = filterWidth - 1 - convInfo.padInfo.left;

  const isChannelsLast = $dataFormat === 'channelsLast';
  const xBatchStride = dx.strides[0];
  const xRowStride = isChannelsLast ? dx.strides[1] : dx.strides[2];
  const xColStride = isChannelsLast ? dx.strides[2] : 1;
  const xChannelStride = isChannelsLast ? 1 : dx.strides[1];
  const yBatchStride = dyStrides[0];
  const yRowStride = isChannelsLast ? dyStrides[1] : dyStrides[2];
  const yColStride = isChannelsLast ? dyStrides[2] : 1;
  const yChannelStride = isChannelsLast ? 1 : dyStrides[1];

  for (let b = 0; b < batchSize; ++b) {
    for (let d1 = 0; d1 < inChannels; ++d1) {
      for (let xR = 0; xR < inHeight; ++xR) {
        const xRCorner = xR - topPad;
        const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
        const yRMax =
            Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);

        for (let xC = 0; xC < inWidth; ++xC) {
          const xCCorner = xC - leftPad;
          const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
          const yCMax =
              Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);

          let dotProd = 0;
          for (let yR = xRMin; yR < yRMax; ++yR) {
            const wR = yR * strideHeight - xRCorner;

            for (let yC = xCMin; yC < yCMax; ++yC) {
              const wC = yC * strideWidth - xCCorner;
              const dyOffset =
                  yBatchStride * b + yRowStride * yR + yColStride * yC;
              const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                  fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

              for (let d2 = 0; d2 < outChannels; ++d2) {
                const pixel = dyValues[dyOffset + yChannelStride * d2];
                const weight = fltValues[fltOffset + d2];
                dotProd += pixel * weight;
              }
            }
          }
          const dxOffset = xBatchStride * b + xRowStride * xR +
              xColStride * xC + xChannelStride * d1;
          dxValues[dxOffset] = dotProd;
        }
      }
    }
  }

  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const conv2DBackpropInputConfig: KernelConfig = {
  kernelName: Conv2DBackpropInput,
  backendName: 'cpu',
  kernelFunc: conv2DBackpropInput as {} as KernelFunc
};
