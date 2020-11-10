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

import {backend_util, Conv3DBackpropFilterV2, Conv3DBackpropFilterV2Attrs, Conv3DBackpropFilterV2Inputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function conv3DBackpropFilterV2(args: {
  inputs: Conv3DBackpropFilterV2Inputs,
  backend: MathBackendCPU,
  attrs: Conv3DBackpropFilterV2Attrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, pad, filterShape} = attrs;

  assertNotComplex([x, dy], 'conv3dBackpropFilterV2');

  const xStrides = util.computeStrides(x.shape);
  const dyStrides = util.computeStrides(dy.shape);

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number], filterShape, strides,
      1 /* dilations */, pad);

  const strideDepth = convInfo.strideDepth;
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const filterDepth = convInfo.filterDepth;
  const filterHeight = convInfo.filterHeight;
  const filterWidth = convInfo.filterWidth;

  const dw = new TensorBuffer(convInfo.filterShape, 'float32');
  const dwValues = dw.values;
  const [dwS0, dwS1, dwS2, dwS3] = dw.strides;
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  const [dyS0, dyS1, dyS2, dyS3] = dyStrides;
  const xValues = backend.data.get(x.dataId).values as TypedArray;
  const [xS0, xS1, xS2, xS3] = xStrides;

  const frontPad = convInfo.padInfo.front;
  const leftPad = convInfo.padInfo.left;
  const topPad = convInfo.padInfo.top;

  for (let wF = 0; wF < filterDepth; ++wF) {
    const yFMin = Math.max(0, Math.ceil((frontPad - wF) / strideDepth));
    const yFMax = Math.min(
        convInfo.outDepth, (convInfo.inDepth + frontPad - wF) / strideDepth);
    const wOffset1 = wF * dwS0;

    for (let wR = 0; wR < filterHeight; ++wR) {
      const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
      const yRMax = Math.min(
          convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
      const wOffset2 = wR * dwS1 + wOffset1;

      for (let wC = 0; wC < filterWidth; ++wC) {
        const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
        const yCMax = Math.min(
            convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
        const wOffset3 = wC * dwS2 + wOffset2;

        for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
          const wOffset4 = d1 * dwS3 + wOffset3;

          for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
            let dotProd = 0;
            for (let b = 0; b < convInfo.batchSize; ++b) {
              const xOffset1 = b * xS0;
              const yOffset1 = b * dyS0;

              for (let yF = yFMin; yF < yFMax; ++yF) {
                const xF = wF + yF * strideDepth - frontPad;
                const xOffset2 = xF * xS1 + xOffset1;
                const yOffset2 = yF * dyS1 + yOffset1;

                for (let yR = yRMin; yR < yRMax; ++yR) {
                  const xR = wR + yR * strideHeight - topPad;
                  const xOffset3 = xR * xS2 + xOffset2;
                  const yOffset3 = yR * dyS2 + yOffset2;

                  for (let yC = yCMin; yC < yCMax; ++yC) {
                    const xC = wC + yC * strideWidth - leftPad;
                    const xOffset4 = xC * xS3 + xOffset3;
                    const yOffset4 = yC * dyS3 + yOffset3;

                    dotProd += xValues[xOffset4 + d1] * dyValues[yOffset4 + d2];
                  }
                }
              }
            }
            dwValues[wOffset4 + d2] = dotProd;
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(dw.shape, dw.dtype, dw.values);
}

export const conv3DBackpropFilterV2Config: KernelConfig = {
  kernelName: Conv3DBackpropFilterV2,
  backendName: 'cpu',
  kernelFunc: conv3DBackpropFilterV2 as {} as KernelFunc
};
