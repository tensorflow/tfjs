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

import {backend_util, Dilation2DAttrs, Dilation2DBackpropFilter, Tensor3D, Tensor4D, TypedArray, util} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export const dilation2dBackpropFilterConfig: KernelConfig = {
  kernelName: Dilation2DBackpropFilter,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x, filter, dy} =
        inputs as {x: Tensor4D, filter: Tensor3D, dy: Tensor4D};
    const {strides, pad, dilations} = attrs as {} as Dilation2DAttrs;
    const cpuBackend = backend as MathBackendCPU;

    const $x =
        util.toNestedArray(
            x.shape, cpuBackend.data.get(x.dataId).values as TypedArray) as
        number[][][][];

    const $filter = util.toNestedArray(
                        filter.shape,
                        cpuBackend.data.get(filter.dataId).values as
                            TypedArray) as number[][][];

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

    util.assert(
        dy.rank === outShape.length,
        () => `Error in ${Dilation2DBackpropFilter}, dy ` +
            `must have the same rank as output ${outShape.length}, but got ` +
            `${dy.rank}`);

    const $dy =
        util.toNestedArray(
            outShape, cpuBackend.data.get(dy.dataId).values as TypedArray) as
        number[][][][];

    // The computed filter gradients has the same dimensions as the filter:
    // [filterHeight, filterWidth, depth]
    const gradients = util.makeZerosNestedTypedArray(
                          filter.shape, filter.dtype) as number[][][];

    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    // This implementation follows the TF c++ implementation:
    // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
    for (let b = 0; b < batchSize; ++b) {
      for (let hOut = 0; hOut < outHeight; ++hOut) {
        const hBeg = hOut * strideHeight - padInfo.top;
        for (let wOut = 0; wOut < outWidth; ++wOut) {
          const wBeg = wOut * strideWidth - padInfo.left;
          for (let d = 0; d < inChannels; ++d) {
            let curVal = Number.MIN_SAFE_INTEGER;
            let hMax = 0;
            let wMax = 0;
            for (let h = 0; h < filterHeight; ++h) {
              const hIn = hBeg + h * dilationHeight;
              if (hIn >= 0 && hIn < inHeight) {
                for (let w = 0; w < filterWidth; ++w) {
                  const wIn = wBeg + w * dilationWidth;
                  if (wIn >= 0 && wIn < inWidth) {
                    const val = $x[b][hIn][wIn][d] + $filter[h][w][d];
                    if (val > curVal) {
                      curVal = val;
                      hMax = h;
                      wMax = w;
                    }
                  }
                }
              }
            }
            gradients[hMax][wMax][d] += $dy[b][hOut][wOut][d];
          }
        }
      }
    }

    const dataId = cpuBackend.write(
        util.toTypedArray(gradients, x.dtype), filter.shape, filter.dtype);

    return {dataId, shape: filter.shape, dtype: filter.dtype};
  }
};
