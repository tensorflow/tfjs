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

import {backend_util, buffer, KernelConfig, KernelFunc, MaxPool3DGrad, MaxPool3DGradAttrs, MaxPool3DGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {maxPool3dPositions} from '../utils/pool_utils';

export function maxPool3DGrad(args: {
  inputs: MaxPool3DGradInputs,
  backend: MathBackendCPU,
  attrs: MaxPool3DGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  assertNotComplex([dy, input], 'maxPool3DGrad');

  const convInfo = backend_util.computePool3DInfo(
      input.shape as [number, number, number, number, number], filterSize,
      strides, 1 /* dilations */, pad, dimRoundingMode);

  const inputBuf = backend.bufferSync(input);
  const maxPosBuf = maxPool3dPositions(inputBuf, convInfo);
  const strideDepth = convInfo.strideDepth;
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationDepth = convInfo.dilationDepth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterDepth = convInfo.effectiveFilterDepth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;
  const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
  const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
  const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
  const dx = buffer(input.shape, 'float32');

  const dyBuf = backend.bufferSync(dy);

  for (let batch = 0; batch < convInfo.batchSize; ++batch) {
    for (let channel = 0; channel < convInfo.inChannels; ++channel) {
      for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
        for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
          for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
            // Shader code begins
            const dyDepthCorner = dxDepth - padFront;
            const dyRowCorner = dxRow - padTop;
            const dyColCorner = dxCol - padLeft;
            let dotProd = 0;
            for (let wDepth = 0; wDepth < effectiveFilterDepth;
                 wDepth += dilationDepth) {
              const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
              if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                  Math.floor(dyDepth) !== dyDepth) {
                continue;
              }
              for (let wRow = 0; wRow < effectiveFilterHeight;
                   wRow += dilationHeight) {
                const dyRow = (dyRowCorner + wRow) / strideHeight;
                if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                    Math.floor(dyRow) !== dyRow) {
                  continue;
                }
                for (let wCol = 0; wCol < effectiveFilterWidth;
                     wCol += dilationWidth) {
                  const dyCol = (dyColCorner + wCol) / strideWidth;
                  if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                      Math.floor(dyCol) !== dyCol) {
                    continue;
                  }

                  const maxPos = effectiveFilterDepth * effectiveFilterHeight *
                          effectiveFilterWidth -
                      1 -
                      (maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel) as
                       number);
                  const curPos =
                      wDepth * effectiveFilterHeight * effectiveFilterWidth +
                      wRow * effectiveFilterWidth + wCol;

                  const mask = maxPos === curPos ? 1 : 0;
                  if (mask === 0) {
                    continue;
                  }

                  const pixel =
                      dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                  dotProd += pixel * mask;
                }
              }
            }
            dx.set(dotProd, batch, dxDepth, dxRow, dxCol, channel);
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}

export const maxPool3DGradConfig: KernelConfig = {
  kernelName: MaxPool3DGrad,
  backendName: 'cpu',
  kernelFunc: maxPool3DGrad as {} as KernelFunc
};
