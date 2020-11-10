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

import {backend_util, buffer, DataType, Rank, TensorBuffer, TypedArray} from '@tensorflow/tfjs-core';

export function pool(
    xValues: TypedArray, xShape: number[], dtype: DataType, strides: number[],
    convInfo: backend_util.Conv2DInfo,
    poolType: 'max'|'avg'): TensorBuffer<Rank, DataType> {
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;
  const padTop = convInfo.padInfo.top;
  const padLeft = convInfo.padInfo.left;

  const initialValue =
      (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                            Number.POSITIVE_INFINITY);

  const output = buffer(convInfo.outShape, dtype);
  const outputVals = output.values;

  const outputBatchStrides =
      convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
  const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
  const outputColStrides = convInfo.outShape[3];

  for (let b = 0; b < convInfo.batchSize; ++b) {
    const outputBatchOffset = b * outputBatchStrides;
    const inputBatchOffset = b * strides[0];
    for (let d = 0; d < convInfo.inChannels; ++d) {
      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
        const xRCorner = yR * strideHeight - padTop;
        const xRMin = Math.max(0, xRCorner);
        const xRMax =
            Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
        const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
        for (let yC = 0; yC < convInfo.outWidth; ++yC) {
          const xCCorner = yC * strideWidth - padLeft;
          const xCMin = Math.max(0, xCCorner);
          const xCMax =
              Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
          let minMaxValue = initialValue;
          let avgValue = 0;
          let count = 0;
          for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
            const xROffset = inputBatchOffset + xR * strides[1];
            for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
              const xCOffset = xROffset + xC * strides[2];
              const pixel = xValues[xCOffset + d];
              if ((poolType === 'max' && pixel > minMaxValue)) {
                minMaxValue = pixel;
              } else if (poolType === 'avg') {
                avgValue += pixel;
                count++;
              }
            }
            if (isNaN(minMaxValue)) {
              break;
            }
          }
          const outputOffset = outputRowOffset + yC * outputColStrides + d;
          outputVals[outputOffset] =
              poolType === 'avg' ? avgValue / count : minMaxValue;
        }
      }
    }
  }
  return output;
}

export function maxPoolPositions(
    xValues: TypedArray, xShape: number[], dtype: DataType,
    convInfo: backend_util.Conv2DInfo, flattenPositions = false,
    includeBatchInIndex = false): TensorBuffer<Rank, 'int32'> {
  const maxPositions = buffer(convInfo.outShape, 'int32');
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;
  const padTop = convInfo.padInfo.top;
  const padLeft = convInfo.padInfo.left;

  const xBuf = buffer(xShape, dtype, xValues);
  for (let b = 0; b < convInfo.batchSize; ++b) {
    for (let d = 0; d < convInfo.inChannels; ++d) {
      for (let yR = 0; yR < convInfo.outHeight; ++yR) {
        const xRCorner = yR * strideHeight - padTop;
        let xRMin = xRCorner;
        while (xRMin < 0) {
          xRMin += dilationHeight;
        }
        // const xRMin = Math.max(0, xRCorner);
        const xRMax =
            Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
        for (let yC = 0; yC < convInfo.outWidth; ++yC) {
          const xCCorner = yC * strideWidth - padLeft;
          let xCMin = xCCorner;
          while (xCMin < 0) {
            xCMin += dilationWidth;
          }
          const xCMax =
              Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
          let maxValue = Number.NEGATIVE_INFINITY;
          let maxPosition = -1;

          for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
            const wR = xR - xRCorner;
            for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
              const wC = xC - xCCorner;
              const pixel = xBuf.get(b, xR, xC, d);
              if (pixel > maxValue) {
                maxValue = pixel as number;
                if (flattenPositions) {
                  maxPosition = includeBatchInIndex ?
                      ((b * convInfo.inHeight + xR) * convInfo.inWidth + xC) *
                              convInfo.inChannels +
                          d :
                      (xR * convInfo.inWidth + xC) * convInfo.inChannels + d;
                } else {
                  maxPosition = wR * effectiveFilterWidth + wC;
                }
              }
            }
          }
          maxPositions.set(maxPosition, b, yR, yC, d);
        }
      }
    }
  }
  return maxPositions;
}
