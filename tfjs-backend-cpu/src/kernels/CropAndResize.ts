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

import {buffer, CropAndResize, CropAndResizeAttrs, CropAndResizeInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function cropAndResize(args: {
  inputs: CropAndResizeInputs,
  backend: MathBackendCPU,
  attrs: CropAndResizeAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image, boxes, boxInd} = inputs;
  const {cropSize, method, extrapolationValue} = attrs;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;
  const numBoxes = boxes.shape[0];

  const [cropHeight, cropWidth] = cropSize;
  const output =
      buffer([numBoxes, cropHeight, cropWidth, numChannels], 'float32');

  const boxVals = backend.data.get(boxes.dataId).values as TypedArray;
  const boxIndVals = backend.data.get(boxInd.dataId).values as TypedArray;
  const imageVals = backend.data.get(image.dataId).values as TypedArray;

  const inStride =
      util.computeStrides(image.shape);  // to calculate flat indexes into image
  const outStride = util.computeStrides(
      output.shape);  // to calculate flat indexes into output

  // Reference implementation
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op.cc
  for (let b = 0; b < numBoxes; b++) {
    const startInd = b * 4;
    const y1 = boxVals[startInd];
    const x1 = boxVals[startInd + 1];
    const y2 = boxVals[startInd + 2];
    const x2 = boxVals[startInd + 3];

    const bInd: number = boxIndVals[b];
    if (bInd >= batch) {
      continue;
    }

    const heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    for (let y = 0; y < cropHeight; y++) {
      const yInd: number = (cropHeight > 1) ?
          y1 * (imageHeight - 1) + y * (heightScale) :
          0.5 * (y1 + y2) * (imageHeight - 1);

      if (yInd < 0 || yInd > imageHeight - 1) {
        for (let x = 0; x < cropWidth; x++) {
          for (let c = 0; c < numChannels; c++) {
            const ind =
                c + x * outStride[2] + y * outStride[1] + b * outStride[0];
            output.values[ind] = extrapolationValue;
          }
        }
        continue;
      }

      if (method === 'bilinear') {
        const topInd = Math.floor(yInd);
        const bottomInd = Math.ceil(yInd);
        const yLerp = yInd - topInd;

        for (let x = 0; x < cropWidth; x++) {
          const xInd = (cropWidth > 1) ?
              x1 * (imageWidth - 1) + x * widthScale :
              0.5 * (x1 + x2) * (imageWidth - 1);

          if (xInd < 0 || xInd > imageWidth - 1) {
            for (let c = 0; c < numChannels; c++) {
              const ind =
                  c + x * outStride[2] + y * outStride[1] + b * outStride[0];
              output.values[ind] = extrapolationValue;
            }
            continue;
          }

          const leftInd = Math.floor(xInd);
          const rightInd = Math.ceil(xInd);
          const xLerp = xInd - leftInd;

          for (let c = 0; c < numChannels; c++) {
            let ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                bInd * inStride[0];
            const topLeft = imageVals[ind];

            ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                bInd * inStride[0];
            const topRight = imageVals[ind];

            ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                bInd * inStride[0];
            const bottomLeft = imageVals[ind];

            ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                bInd * inStride[0];
            const bottomRight = imageVals[ind];

            const top = topLeft + (topRight - topLeft) * xLerp;
            const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;

            ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
            output.values[ind] = top + ((bottom - top) * yLerp);
          }
        }
      } else {  // method == "nearest"
        for (let x = 0; x < cropWidth; ++x) {
          const xInd = (cropWidth > 1) ?
              x1 * (imageWidth - 1) + x * widthScale :
              0.5 * (x1 + x2) * (imageWidth - 1);

          if (xInd < 0 || xInd > imageWidth - 1) {
            for (let c = 0; c < numChannels; c++) {
              const ind =
                  c + x * outStride[2] + y * outStride[1] + b * outStride[0];
              output.values[ind] = extrapolationValue;
            }
            continue;
          }

          const closestX = Math.round(xInd);
          const closestY = Math.round(yInd);
          for (let c = 0; c < numChannels; c++) {
            const inInd = c + closestX * inStride[2] + closestY * inStride[1] +
                bInd * inStride[0];
            const outInd =
                c + x * outStride[2] + y * outStride[1] + b * outStride[0];
            output.values[outInd] = imageVals[inInd];
          }
        }
      }
    }
  }

  return backend.makeTensorInfo(output.shape, output.dtype, output.values);
}

export const cropAndResizeConfig: KernelConfig = {
  kernelName: CropAndResize,
  backendName: 'cpu',
  kernelFunc: cropAndResize as {} as KernelFunc
};
