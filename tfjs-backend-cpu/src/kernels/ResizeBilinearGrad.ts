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

import {KernelConfig, KernelFunc, ResizeBilinearGrad, ResizeBilinearGradAttrs, ResizeBilinearGradInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function resizeBilinearGrad(args: {
  inputs: ResizeBilinearGradInputs,
  backend: MathBackendCPU,
  attrs: ResizeBilinearGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images, dy} = inputs;
  const {alignCorners} = attrs;

  assertNotComplex([dy, images], 'resizeBilinearGrad');

  const imagesStrides = util.computeStrides(images.shape);

  const [batch, xHeight, xWidth, depth] = images.shape;
  const [, yHeight, yWidth] = dy.shape;

  const output = new Float32Array(batch * xHeight * xWidth * depth);

  // In the backwards pass, we want to find the pixels that were generated
  // for each pixel in the input image the forward pass and add the
  // corresponding coefficient from dy to the gradient (with some
  // interpolation).

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

  // Reference implementation
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275
  const dyValues = backend.data.get(dy.dataId).values as TypedArray;
  let offset = 0;
  for (let b = 0; b < batch; b++) {
    const bOffset = b * imagesStrides[0];
    for (let r = 0; r < yHeight; r++) {
      const dxR = r * heightScale;
      const topDxRIndex = Math.floor(dxR);
      const bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);

      const topDxROffset = bOffset + topDxRIndex * imagesStrides[1];
      const bottomDxROffset = bOffset + bottomDxRIndex * imagesStrides[1];

      const dxRLerp = dxR - topDxRIndex;
      const inverseDxRLerp = 1.0 - dxRLerp;
      for (let c = 0; c < yWidth; c++) {
        const dxC = c * widthScale;
        const leftDxCIndex = Math.floor(dxC);
        const rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
        const dxCLerp = dxC - leftDxCIndex;
        const inverseDxCLerp = 1.0 - dxCLerp;

        const topLeftRCOffset = topDxROffset + leftDxCIndex * imagesStrides[2];
        const topRightRCOffset =
            topDxROffset + rightDxCIndex * imagesStrides[2];
        const bottomLeftRCOffset =
            bottomDxROffset + leftDxCIndex * imagesStrides[2];
        const bottomRightRCOffset =
            bottomDxROffset + rightDxCIndex * imagesStrides[2];

        const inverseDxRLerpTimesInverseDxCLerp =
            inverseDxRLerp * inverseDxCLerp;
        const inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
        const dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
        const dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
        for (let d = 0; d < depth; d++) {
          const dyVal = dyValues[offset++];
          output[topLeftRCOffset + d] +=
              dyVal * inverseDxRLerpTimesInverseDxCLerp;
          output[topRightRCOffset + d] += dyVal * inverseDxRLerpTimesDxCLerp;
          output[bottomLeftRCOffset + d] += dyVal * dxRLerpTimesInverseDxCLerp;
          output[bottomRightRCOffset + d] += dyVal * dxRLerpTimesDxCLerp;
        }
      }
    }
  }

  return backend.makeTensorInfo(
      [batch, xWidth, xHeight, depth], 'float32', output);
}

export const resizeBilinearGradConfig: KernelConfig = {
  kernelName: ResizeBilinearGrad,
  backendName: 'cpu',
  kernelFunc: resizeBilinearGrad as {} as KernelFunc
};
