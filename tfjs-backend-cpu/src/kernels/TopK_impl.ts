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

/** An implementation of the TopK kernel shared between webgl and cpu. */

import {buffer, NumericDataType, Rank, ShapeMap, Tensor, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

export function topKImpl<T extends Tensor, R extends Rank>(
    x: TypedArray, xShape: number[], xDtype: NumericDataType, k: number,
    sorted: boolean):
    [TensorBuffer<R, NumericDataType>, TensorBuffer<R, 'int32'>] {
  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const lastDim = xShape[xShape.length - 1];
  const [batch, size] = [x.length / lastDim, lastDim];
  const allTopKVals = util.getTypedArrayFromDType(xDtype, batch * k);
  const allTopKIndices = util.getTypedArrayFromDType('int32', batch * k);

  for (let b = 0; b < batch; b++) {
    const offset = b * size;
    const vals = x.subarray(offset, offset + size);
    const valAndInd: Array<{value: number, index: number}> = [];
    for (let i = 0; i < vals.length; i++) {
      valAndInd.push({value: vals[i], index: i});
    }
    valAndInd.sort((a, b) => b.value - a.value);

    const outOffset = b * k;
    const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
    const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
    for (let i = 0; i < k; i++) {
      topKVals[i] = valAndInd[i].value;
      topKIndices[i] = valAndInd[i].index;
    }
  }
  // Reshape back to the original input shape, except that the last
  // dimension is k.
  const outputShape = xShape.slice();
  outputShape[outputShape.length - 1] = k;

  return [
    buffer(outputShape as ShapeMap[R], xDtype, allTopKVals),
    buffer(outputShape as ShapeMap[R], 'int32', allTopKIndices)
  ];
}
