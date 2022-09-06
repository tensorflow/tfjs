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

import {backend_util, Conv2DBackpropFilter, Conv2DBackpropFilterAttrs, Conv2DBackpropFilterInputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function conv2DBackpropFilter(args: {
  inputs: Conv2DBackpropFilterInputs,
  backend: MathBackendCPU,
  attrs: Conv2DBackpropFilterAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, pad, dataFormat, dimRoundingMode, filterShape} = attrs;

  assertNotComplex([x, dy], 'conv2dBackpropFilter');

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number], filterShape, strides,
      1 /* dilations */, pad, dimRoundingMode, false /* depthwise */,
      $dataFormat);

  const {strideHeight, strideWidth, filterHeight, filterWidth} = convInfo;
  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  const dW = new TensorBuffer(convInfo.filterShape, 'float32');

  const leftPad = convInfo.padInfo.left;
  const topPad = convInfo.padInfo.top;
  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const dyVals = backend.data.get(dy.dataId).values as TypedArray;

  const xBuf = new TensorBuffer(x.shape, x.dtype, xVals);
  const dyBuf = new TensorBuffer(dy.shape, dy.dtype, dyVals);

  for (let wR = 0; wR < filterHeight; ++wR) {
    const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
    const yRMax = Math.min(
        convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);

    for (let wC = 0; wC < filterWidth; ++wC) {
      const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
      const yCMax = Math.min(
          convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);

      for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
        for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
          let dotProd = 0;
          for (let b = 0; b < convInfo.batchSize; ++b) {
            for (let yR = yRMin; yR < yRMax; ++yR) {
              const xR = wR + yR * strideHeight - topPad;
              for (let yC = yCMin; yC < yCMax; ++yC) {
                const xC = wC + yC * strideWidth - leftPad;
                if (isChannelsLast) {
                  dotProd += (xBuf.get(b, xR, xC, d1) as number) *
                      (dyBuf.get(b, yR, yC, d2) as number);
                } else {
                  dotProd += (xBuf.get(b, d1, xR, xC) as number) *
                      (dyBuf.get(b, d2, yR, yC) as number);
                }
              }
            }
          }
          dW.set(dotProd, wR, wC, d1, d2);
        }
      }
    }
  }

  return backend.makeTensorInfo(dW.shape, dW.dtype, dW.values);
}

export const conv2DBackpropFilterConfig: KernelConfig = {
  kernelName: Conv2DBackpropFilter,
  backendName: 'cpu',
  kernelFunc: conv2DBackpropFilter as {} as KernelFunc
};
