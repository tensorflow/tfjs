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

import {KernelConfig, KernelFunc, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function resizeNearestNeighborGrad(args: {
  inputs: ResizeNearestNeighborGradInputs,
  backend: MathBackendCPU,
  attrs: ResizeNearestNeighborGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images, dy} = inputs;
  const {alignCorners} = attrs;

  assertNotComplex([dy, images], 'resizeNearestNeighborGrad');

  const imagesStrides = util.computeStrides(images.shape);
  const dyStrides = util.computeStrides(dy.shape);
  const [batch, xHeight, xWidth, depth] = images.shape;
  const [, yHeight, yWidth] = dy.shape;

  const output = new Float32Array(batch * xHeight * xWidth * depth);
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;

  // In the backwards pass, we want to find the pixels that were generated
  // for each pixel in the input image the forward pass

  const effectiveXSize: [number, number] = [
    (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
    (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
  ];

  const effectiveYSize: [number, number] = [
    (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
    (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
  ];

  const heightScale = effectiveXSize[0] / effectiveYSize[0];
  const widthScale = effectiveXSize[1] / effectiveYSize[1];

  const invHeightScale = 1 / heightScale;
  const invWidthScale = 1 / widthScale;

  // This defines the size of the window of values around a particular
  // index in dy that we want to search for contributions to dx.
  const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
  const winWidth = (Math.ceil(invWidthScale) * 2) + 2;

  // Loop over the output space.
  for (let b = 0; b < batch; b++) {
    const batchOffset = b * imagesStrides[0];
    for (let r = 0; r < xHeight; r++) {
      const rowOffset = batchOffset + r * imagesStrides[1];

      // Compute bounds for where in dy we will look
      const startRLerp = Math.floor(r * invHeightScale);
      const startDyR = Math.floor(startRLerp - (winHeight / 2));
      for (let c = 0; c < xWidth; c++) {
        const colOffset = rowOffset + c * imagesStrides[2];

        // Compute bounds for where in dy we will look
        const startCLerp = Math.floor(c * invWidthScale);
        const startDyC = Math.floor(startCLerp - (winWidth / 2));

        for (let d = 0; d < depth; d++) {
          let accum = 0;
          // loop over dy

          for (let dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
            const dyR = dyRIndex + startDyR;
            // Guard against the window exceeding the bounds of dy
            if (dyR < 0 || dyR >= yHeight) {
              continue;
            }

            const dyROffset = batchOffset + dyR * dyStrides[1];
            const sourceFracRow = dyR * heightScale;
            const sourceNearestRow = Math.min(
                xHeight - 1,
                alignCorners ? Math.round(sourceFracRow) :
                               Math.floor(sourceFracRow));
            if (r !== sourceNearestRow) {
              continue;
            }
            for (let dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
              const dyC = dyCIndex + startDyC;
              // Guard against the window exceeding the bounds of dy
              if (dyC < 0 || dyC >= yWidth) {
                continue;
              }

              const dyCOffset = dyROffset + dyC * dyStrides[2];
              const sourceFracCol = dyC * widthScale;
              const sourceNearestCol = Math.min(
                  xWidth - 1,
                  alignCorners ? Math.round(sourceFracCol) :
                                 Math.floor(sourceFracCol));

              if (c === sourceNearestCol) {
                accum += dyValues[dyCOffset + d];
              }
            }
          }
          output[colOffset + d] = accum;
        }
      }
    }
  }

  return backend.makeTensorInfo(images.shape, images.dtype, output);
}

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'cpu',
  kernelFunc: resizeNearestNeighborGrad as {} as KernelFunc
};
