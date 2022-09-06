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

import {KernelConfig, KernelFunc, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function resizeBilinear(args: {
  inputs: ResizeBilinearInputs,
  backend: MathBackendCPU,
  attrs: ResizeBilinearAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, halfPixelCenters, size} = attrs;

  assertNotComplex(images, 'resizeBilinear');

  const imagesStrides = util.computeStrides(images.shape);
  const [newHeight, newWidth] = size;

  const [batch, oldHeight, oldWidth, numChannels] = images.shape;
  const xValues = backend.data.get(images.dataId).values as TypedArray;
  const result = new Float32Array(
      util.sizeFromShape([batch, newHeight, newWidth, numChannels]));

  const effectiveInputSize: [number, number] = [
    (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
    (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
  ];

  const effectiveOutputSize: [number, number] = [
    (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
    (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
  ];
  let outputIdx = 0;
  const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
  const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
  for (let b = 0; b < batch; b++) {
    for (let r = 0; r < newHeight; r++) {
      let sourceFracRow: number;
      if (halfPixelCenters) {
        sourceFracRow = effectiveRowSizeRatio * (r + 0.5) - 0.5;
      } else {
        sourceFracRow = effectiveRowSizeRatio * r;
      }

      const sourceRowFloor = Math.max(0, Math.floor(sourceFracRow));
      const rowFrac = sourceFracRow - sourceRowFloor;
      const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
      const topRowOffset =
          b * imagesStrides[0] + sourceRowFloor * imagesStrides[1];
      const botRowOffset =
          b * imagesStrides[0] + sourceRowCeil * imagesStrides[1];
      for (let c = 0; c < newWidth; c++) {
        let sourceFracCol: number;
        if (halfPixelCenters) {
          sourceFracCol = effectiveColSizeRatio * (c + 0.5) - 0.5;
        } else {
          sourceFracCol = effectiveColSizeRatio * c;
        }
        const sourceColFloor = Math.max(0, Math.floor(sourceFracCol));
        const colFrac = sourceFracCol - sourceColFloor;
        const sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
        const topLeftOffest = topRowOffset + sourceColFloor * imagesStrides[2];
        const botLeftOffset = botRowOffset + sourceColFloor * imagesStrides[2];
        const topRightOffset = topRowOffset + sourceColCeil * imagesStrides[2];
        const botRightOffest = botRowOffset + sourceColCeil * imagesStrides[2];
        for (let d = 0; d < numChannels; d++) {
          // Begin shader.

          // Compute the fractional index of the source.
          const topLeft = xValues[topLeftOffest + d];
          const bottomLeft = xValues[botLeftOffset + d];
          const topRight = xValues[topRightOffset + d];
          const bottomRight = xValues[botRightOffest + d];

          const top = topLeft + (topRight - topLeft) * colFrac;
          const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
          const newValue = top + (bottom - top) * rowFrac;

          result[outputIdx++] = newValue;
        }
      }
    }
  }

  return backend.makeTensorInfo(
      [batch, newHeight, newWidth, numChannels], 'float32', result);
}

export const resizeBilinearConfig: KernelConfig = {
  kernelName: ResizeBilinear,
  backendName: 'cpu',
  kernelFunc: resizeBilinear as {} as KernelFunc
};
