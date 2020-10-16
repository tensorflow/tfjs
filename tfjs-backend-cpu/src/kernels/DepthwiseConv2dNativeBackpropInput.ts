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

import {backend_util, DepthwiseConv2dNativeBackpropInput, DepthwiseConv2dNativeBackpropInputAttrs, DepthwiseConv2dNativeBackpropInputInputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function depthwiseConv2dNativeBackpropInput(args: {
  inputs: DepthwiseConv2dNativeBackpropInputInputs,
  backend: MathBackendCPU,
  attrs: DepthwiseConv2dNativeBackpropInputAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {strides, dilations, pad, dimRoundingMode, inputShape} = attrs;

  assertNotComplex([dy, filter], 'depthwiseConv2DNativeBackpropInput');

  const dyStrides = util.computeStrides(dy.shape);
  const filterStrides = util.computeStrides(filter.shape);

  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      dilations, pad, dimRoundingMode, true /* depthwise */);

  const dx = new TensorBuffer(convInfo.inShape, 'float32');
  const dxValues = dx.values;
  const [dxS0, dxS1, dxS2] = dx.strides;
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  const [dyS0, dyS1, dyS2] = dyStrides;
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
  const topPad = filterHeight - 1 - convInfo.padInfo.top;
  const leftPad = filterWidth - 1 - convInfo.padInfo.left;
  const chMul = outChannels / inChannels;

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
              const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
              const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                  fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;

              for (let dm = 0; dm < chMul; ++dm) {
                const d2 = d1 * chMul + dm;
                const pixel = dyValues[dyOffset + d2];
                const weight = fltValues[fltOffset + dm];
                dotProd += pixel * weight;
              }
            }
          }
          dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
        }
      }
    }
  }

  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const depthwiseConv2dNativeBackpropInputConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropInput,
  backendName: 'cpu',
  kernelFunc: depthwiseConv2dNativeBackpropInput as {} as KernelFunc
};
