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

import {backend_util, Dilation2D, Dilation2DAttrs, Dilation2DInputs, KernelConfig, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const dilation2dConfig: KernelConfig = {
  kernelName: Dilation2D,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter} = inputs as Dilation2DInputs;
    const {strides, pad, dilations} = attrs as {} as Dilation2DAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const xVals = cpuBackend.data.get(x.dataId).values as TypedArray;
    const xRank = x.shape.length;

    const filterVals = cpuBackend.data.get(filter.dataId).values as TypedArray;
    const filterRank = filter.shape.length;

    const {
      batchSize,
      inHeight,
      inWidth,
      inChannels,
      outHeight,
      outWidth,
      padInfo,
      strideHeight,
      strideWidth,
      filterHeight,
      filterWidth,
      dilationHeight,
      dilationWidth,
      outShape
    } =
        backend_util.computeDilation2DInfo(
            x.shape as [number, number, number, number],
            filter.shape as [number, number, number], strides, pad,
            'NHWC' /* dataFormat */, dilations);

    const outSize = util.sizeFromShape(outShape);
    const outRank = outShape.length;
    const outputVals = util.getArrayFromDType(x.dtype, outSize);

    // Upsampling the input by fill in `dilation size - 1` values between each
    // input value.
    // This implementation follows the TF c++ implementation:
    // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
    for (let b = 0; b < batchSize; ++b) {
      for (let hOut = 0; hOut < outHeight; ++hOut) {
        const hBeg = hOut * strideHeight - padInfo.top;
        for (let wOut = 0; wOut < outWidth; ++wOut) {
          const wBeg = wOut * strideWidth - padInfo.left;
          for (let d = 0; d < inChannels; ++d) {
            let curVal = Number.MIN_SAFE_INTEGER;
            for (let h = 0; h < filterHeight; ++h) {
              const hIn = hBeg + h * dilationHeight;
              if (hIn >= 0 && hIn < inHeight) {
                for (let w = 0; w < filterWidth; ++w) {
                  const wIn = wBeg + w * dilationWidth;
                  if (wIn >= 0 && wIn < inWidth) {
                    const xIndex = util.locToIndex(
                        [b, hIn, wIn, d], xRank, util.computeStrides(x.shape));
                    const filterIndex = util.locToIndex(
                        [h, w, d], filterRank,
                        util.computeStrides(filter.shape));
                    const val = xVals[xIndex] + filterVals[filterIndex];
                    if (val > curVal) {
                      curVal = val;
                    }
                  }
                }
              }
            }
            const outputIndex = util.locToIndex(
                [b, hOut, wOut, d], outRank, util.computeStrides(outShape));
            outputVals[outputIndex] = curVal;
          }
        }
      }
    }

    const dataId = cpuBackend.write(
        util.toTypedArray(outputVals, x.dtype), outShape, x.dtype);

    return {dataId, shape: outShape, dtype: x.dtype};
  }
};
