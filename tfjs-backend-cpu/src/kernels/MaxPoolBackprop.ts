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
import {backend_util, buffer, KernelConfig, KernelFunc, MaxPoolBackprop, MaxPoolBackpropAttrs, MaxPoolBackpropInputs, Rank, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {maxPoolPositions} from '../utils/pool_utils';

export function maxPoolBackprop(args: {
  inputs: MaxPoolBackpropInputs,
  backend: MathBackendCPU,
  attrs: MaxPoolBackpropAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input, output} = inputs;
  const x = input;
  assertNotComplex([input, output], 'maxPoolBackprop');
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad, dimRoundingMode);
  const xValues = backend.data.get(x.dataId).values as TypedArray;
  const maxPosBuf = buffer(
      convInfo.outShape, x.dtype,
      maxPoolPositions(xValues, x.shape, x.dtype, convInfo).values);
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;
  const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
  const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
  const dx =
      buffer<Rank.R4>(x.shape as [number, number, number, number], 'float32');

  const dyData = backend.data.get(dy.dataId).values as Float32Array;
  const dyBuf = buffer<Rank.R4>(
      dy.shape as [number, number, number, number], 'float32', dyData);

  for (let b = 0; b < convInfo.batchSize; ++b) {
    for (let d = 0; d < convInfo.inChannels; ++d) {
      for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
        for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
          // Shader code begins.
          const dyRCorner = dxR - padTop;
          const dyCCorner = dxC - padLeft;
          let dotProd = 0;
          for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
            const dyR = (dyRCorner + wR) / strideHeight;
            if (dyR < 0 || dyR >= convInfo.outHeight ||
                Math.floor(dyR) !== dyR) {
              continue;
            }
            for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
              const dyC = (dyCCorner + wC) / strideWidth;
              if (dyC < 0 || dyC >= convInfo.outWidth ||
                  Math.floor(dyC) !== dyC) {
                continue;
              }
              const maxPos = effectiveFilterHeight * effectiveFilterWidth - 1 -
                  (maxPosBuf.get(b, dyR, dyC, d) as number);
              const curPos = wR * effectiveFilterWidth + wC;

              const mask = maxPos === curPos ? 1 : 0;
              if (mask === 0) {
                continue;
              }

              const pixel = dyBuf.get(b, dyR, dyC, d);
              dotProd += pixel * mask;
            }
          }
          dx.set(dotProd, b, dxR, dxC, d);
        }
      }
    }
  }
  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const maxPoolBackpropConfig: KernelConfig = {
  kernelName: MaxPoolBackprop,
  backendName: 'cpu',
  kernelFunc: maxPoolBackprop as {} as KernelFunc
};
