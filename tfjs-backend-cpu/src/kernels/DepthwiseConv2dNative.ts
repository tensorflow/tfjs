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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, KernelFunc, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function depthwiseConv2dNative(args: {
  inputs: DepthwiseConv2dNativeInputs,
  backend: MathBackendCPU,
  attrs: DepthwiseConv2dNativeAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations, dimRoundingMode} = attrs;

  assertNotComplex([x, filter], 'depthwiseConv2DNative');

  const xStrides = util.computeStrides(x.shape);
  const filterStrides = util.computeStrides(filter.shape);

  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, $dilations),
      () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
          `1. Got strides ${strides} and dilations '${$dilations}'`);

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */);

  const {filterHeight, filterWidth, dilationHeight, dilationWidth, padInfo} =
      convInfo;
  const padLeft = padInfo.left;
  const padTop = padInfo.top;
  const chMul = convInfo.outChannels / convInfo.inChannels;
  const y = new TensorBuffer(convInfo.outShape, x.dtype as 'float32');
  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const wVals = backend.data.get(filter.dataId).values as TypedArray;
  const yVals = y.values;

  for (let b = 0; b < convInfo.batchSize; ++b) {
    const xOffset1 = b * xStrides[0];
    const yOffset1 = b * y.strides[0];
    for (let yR = 0; yR < convInfo.outHeight; ++yR) {
      const yOffset2 = yOffset1 + yR * y.strides[1];
      const xRCorner = yR * convInfo.strideHeight - padTop;
      for (let wR = 0; wR < filterHeight; ++wR) {
        const xR = xRCorner + wR * dilationHeight;
        if (xR < 0 || xR >= convInfo.inHeight) {
          continue;
        }
        const wOffset1 = wR * filterStrides[0];
        const xOffset2 = xOffset1 + xR * xStrides[1];
        for (let yC = 0; yC < convInfo.outWidth; ++yC) {
          const yOffset3 = yOffset2 + yC * y.strides[2];
          const xCCorner = yC * convInfo.strideWidth - padLeft;
          for (let wC = 0; wC < filterWidth; ++wC) {
            const xC = xCCorner + wC * dilationWidth;
            if (xC < 0 || xC >= convInfo.inWidth) {
              continue;
            }
            const wOffset2 = wOffset1 + wC * filterStrides[1];
            const xOffset3 = xOffset2 + xC * convInfo.inChannels;
            let yOffset4 = yOffset3;
            let wOffset3 = wOffset2;
            for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
              const xVal = xVals[xOffset3 + d1];
              for (let q = 0; q < chMul; ++q) {
                yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
              }
              yOffset4 += chMul;
              wOffset3 += chMul;
            }
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(y.shape, y.dtype, y.values);
}

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'cpu',
  kernelFunc: depthwiseConv2dNative as {} as KernelFunc
};
