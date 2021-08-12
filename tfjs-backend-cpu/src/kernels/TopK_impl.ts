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

type Pair = {
  value: number,
  index: number
};

const comparePair = (a: Pair, b: Pair) => {
  const valueDiff = b.value - a.value;
  return valueDiff === 0 ? a.index - b.index : valueDiff;
};

/**
 * Partitions array where all elements smaller than the (k+1) smallest element
 * are found to the left of it, and all larger to the right of it.
 * Based on the Floyd-Rivest Algorithm, ref:
 * https://en.wikipedia.org/wiki/Floyd%E2%80%93Rivest_algorithm
 * @param array: Array to partition
 * @param left: Left index for the interval
 * @param right: Right index for the interval
 * @param k: Desired index value, where array[k] is the (k+1)th smallest element
 *           when left = 0
 */
function select(array: Pair[], k: number, left = 0, right = array.length - 1) {
  while (right > left) {
    // Use select recursively to sample a smaller set of size s
    // the arbitrary constants 600 and 0.5 are used in the original
    // version to minimize execution time.
    if (right - left > 600) {
      const n = right - left + 1;
      const i = k - left + 1;
      const z = Math.log(n);
      const s = 0.5 * Math.exp(2 * z / 3);
      const sd = 0.5 * Math.sqrt(z * s * (n - s) / n) * Math.sign(i - n / 2);
      const newLeft = Math.max(left, Math.floor(k - i * s / n + sd));
      const newRight = Math.min(right, Math.floor(k + (n - i) * s / n + sd));
      select(array, k, newLeft, newRight);
    }
    // partition the elements between left and right around t
    const t = array[k];
    let i = left;
    let j = right;

    util.swap(array, left, k);

    if (comparePair(array[right], t) > 0) {
      util.swap(array, left, right);
    }
    while (i < j) {
      util.swap(array, i, j);
      i++;
      j--;
      while (comparePair(array[i], t) < 0) {
        i = i + 1;
      }
      while (comparePair(array[j], t) > 0) {
        j = j - 1;
      }
    }
    if (comparePair(array[left], t) === 0) {
      util.swap(array, left, j);
    } else {
      j = j + 1;
      util.swap(array, j, right);
    }
    // Adjust left and right towards the boundaries of the subset
    // containing the (k - left + 1)th smallest element.
    if (j <= k) {
      left = j + 1;
    }
    if (k <= j) {
      right = j - 1;
    }
  }
}

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

    let valAndInd: Pair[] = new Array(vals.length);
    vals.forEach(
        (value: number, index: number) => valAndInd[index] = {value, index});

    if (k < valAndInd.length) {
      select(valAndInd, k);
      valAndInd = valAndInd.slice(0, k);
    }

    if (sorted) {
      valAndInd.sort(comparePair);
    }
    
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
