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

import {backend_util, Conv3DBackpropInputV2, Conv3DBackpropInputV2Attrs, Conv3DBackpropInputV2Inputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function conv3DBackpropInputV2(args: {
  inputs: Conv3DBackpropInputV2Inputs,
  backend: MathBackendCPU,
  attrs: Conv3DBackpropInputV2Attrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {pad, strides, inputShape} = attrs;

  assertNotComplex([dy], 'conv3dBackpropInputV2');

  const dyStrides = util.computeStrides(dy.shape);
  const filterStrides = util.computeStrides(filter.shape);

  const convInfo = backend_util.computeConv3DInfo(
      inputShape, filter.shape as [number, number, number, number, number],
      strides, 1 /* dilations */, pad);

  const dx = new TensorBuffer(convInfo.inShape, 'float32');
  const dxValues = dx.values;
  const [dxS0, dxS1, dxS2, dxS3] = dx.strides;
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  const [dyS0, dyS1, dyS2, dyS3] = dyStrides;
  const fltValues = backend.data.get(filter.dataId).values as TypedArray;
  const [fltS0, fltS1, fltS2, fltS3] = filterStrides;
  const {
    batchSize,
    filterDepth,
    filterHeight,
    filterWidth,
    inChannels,
    inDepth,
    inHeight,
    inWidth,
    outChannels,
    outDepth,
    outHeight,
    outWidth,
    strideDepth,
    strideHeight,
    strideWidth
  } = convInfo;
  const frontPad = filterDepth - 1 - convInfo.padInfo.front;
  const topPad = filterHeight - 1 - convInfo.padInfo.top;
  const leftPad = filterWidth - 1 - convInfo.padInfo.left;

  for (let b = 0; b < batchSize; ++b) {
    for (let d1 = 0; d1 < inChannels; ++d1) {
      // Frames of depth
      for (let xF = 0; xF < inDepth; ++xF) {
        const xFCorner = xF - frontPad;
        const xFMin = Math.max(0, Math.ceil(xFCorner / strideDepth));
        const yFMax =
            Math.min(outDepth, (filterDepth + xFCorner) / strideDepth);

        // Rows as per standard 2d matrix notation
        for (let xR = 0; xR < inHeight; ++xR) {
          const xRCorner = xR - topPad;
          const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
          const yRMax =
              Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
          // Columns as per standard 2d matrix notation
          for (let xC = 0; xC < inWidth; ++xC) {
            const xCCorner = xC - leftPad;
            const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
            const yCMax =
                Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);

            let dotProd = 0;
            for (let yF = xFMin; yF < yFMax; ++yF) {
              const wF = yF * strideDepth - xFCorner;

              for (let yR = xRMin; yR < yRMax; ++yR) {
                const wR = yR * strideHeight - xRCorner;

                for (let yC = xCMin; yC < yCMax; ++yC) {
                  const wC = yC * strideWidth - xCCorner;
                  const dyOffset = dyS0 * b + dyS1 * yF + dyS2 * yR + dyS3 * yC;
                  const fltOffset = fltS0 * (filterDepth - 1 - wF) +
                      fltS1 * (filterHeight - 1 - wR) +
                      fltS2 * (filterWidth - 1 - wC) + fltS3 * d1;

                  for (let d2 = 0; d2 < outChannels; ++d2) {
                    const pixel = dyValues[dyOffset + d2];
                    const weight = fltValues[fltOffset + d2];
                    dotProd += pixel * weight;
                  }
                }
              }
            }
            dxValues[dxS0 * b + dxS1 * xF + dxS2 * xR + dxS3 * xC + d1] =
                dotProd;
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const conv3DBackpropInputV2Config: KernelConfig = {
  kernelName: Conv3DBackpropInputV2,
  backendName: 'cpu',
  kernelFunc: conv3DBackpropInputV2 as {} as KernelFunc
};
