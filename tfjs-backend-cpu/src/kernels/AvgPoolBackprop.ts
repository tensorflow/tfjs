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
import {AvgPoolBackprop, AvgPoolBackpropAttrs, AvgPoolBackpropInputs, backend_util, buffer, KernelConfig, KernelFunc, Rank, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function avgPoolBackprop(args: {
  inputs: AvgPoolBackpropInputs,
  backend: MathBackendCPU,
  attrs: AvgPoolBackpropAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  assertNotComplex([dy, input], 'avgPoolBackprop');
  const {filterSize, strides, pad} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad);
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const filterHeight = convInfo.filterHeight;
  const filterWidth = convInfo.filterWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;
  const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
  const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
  const dx =
      buffer<Rank.R4>(x.shape as [number, number, number, number], 'float32');

  const avgMultiplier = 1 / (filterHeight * filterWidth);

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

              const pixel = dyBuf.get(b, dyR, dyC, d);
              dotProd += pixel;
            }
          }
          dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
        }
      }
    }
  }
  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const avgPoolBackpropConfig: KernelConfig = {
  kernelName: AvgPoolBackprop,
  backendName: 'cpu',
  kernelFunc: avgPoolBackprop as {} as KernelFunc
};
