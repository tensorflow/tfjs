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

import {KernelConfig, KernelFunc, ResizeNearestNeighbor, ResizeNearestNeighborAttrs, ResizeNearestNeighborInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function resizeNearestNeighbor(args: {
  inputs: ResizeNearestNeighborInputs,
  backend: MathBackendCPU,
  attrs: ResizeNearestNeighborAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, halfPixelCenters, size} = attrs;

  assertNotComplex(images, 'resizeNearestNeighbor');

  const imagesStrides = util.computeStrides(images.shape);
  const [newHeight, newWidth] = size;

  const [batch, oldHeight, oldWidth, numChannels] = images.shape;
  const xValues = backend.data.get(images.dataId).values as TypedArray;
  const output = new Float32Array(batch * newHeight * newWidth * numChannels);

  const effectiveInputSize: [number, number] = [
    (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
    (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
  ];

  const effectiveOutputSize: [number, number] = [
    (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
    (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
  ];

  const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
  const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];

  let outputOffset = 0;
  for (let b = 0; b < batch; b++) {
    const batchOffset = b * imagesStrides[0];
    for (let r = 0; r < newHeight; r++) {
      const sourceFracRow = halfPixelCenters ?
          effectiveRowSizeRatio * (r + 0.5) :
          effectiveRowSizeRatio * r;
      let sourceNearestRow = Math.min(
          oldHeight - 1,
          alignCorners ? Math.round(sourceFracRow) : Math.floor(sourceFracRow));
      if (halfPixelCenters) {
        sourceNearestRow = Math.max(0, sourceNearestRow);
      }
      const rowOffset = batchOffset + sourceNearestRow * imagesStrides[1];
      for (let c = 0; c < newWidth; c++) {
        const sourceFracCol = halfPixelCenters ?
            effectiveColSizeRatio * (c + 0.5) :
            effectiveColSizeRatio * c;
        let sourceNearestCol = Math.min(
            oldWidth - 1,
            alignCorners ? Math.round(sourceFracCol) :
                           Math.floor(sourceFracCol));
        if (halfPixelCenters) {
          sourceNearestCol = Math.max(0, sourceNearestCol);
        }
        const colOffset = rowOffset + sourceNearestCol * imagesStrides[2];
        for (let d = 0; d < numChannels; d++) {
          // Begin shader.
          // Compute the fractional index of the source.
          const newVal = xValues[colOffset + d];
          output[outputOffset++] = newVal;
        }
      }
    }
  }

  return backend.makeTensorInfo(
      [batch, newHeight, newWidth, numChannels], images.dtype, output);
}

export const resizeNearestNeighborConfig: KernelConfig = {
  kernelName: ResizeNearestNeighbor,
  backendName: 'cpu',
  kernelFunc: resizeNearestNeighbor as {} as KernelFunc
};
