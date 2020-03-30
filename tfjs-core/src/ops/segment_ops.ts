/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {Tensor, Tensor1D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, isInt} from '../util';
import {expandDims} from './array_ops';
import {maximum} from './binary_ops';
import {greaterEqual} from './compare';
import {gather} from './gather';
import {logicalAnd, where} from './logical_ops';
import {op} from './operation';
import {ones, scalar, zerosLike} from './tensor_ops';

/**
 * Computes the sum along segments of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
 * const numSegments = 3;
 *
 * x.unsortedSegmentSum(segmentIds, numSegments).print()
 * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
 * ```
 * @param x The `tf.Tensor` that will be summed along its segments.
 * @param segmentIds A `tf.Tensor1D` whose rank is equal to the rank of `x`'s
 * dimension along the `axis`.  Maps each element of `x` to a segment.
 * @param numSegments The number of distinct `segmentIds`.
 */
/** @doc {heading: 'Operations', subheading: 'Segment'} */
function unsortedSegmentSum_<T extends Tensor>(
    x: T|TensorLike, segmentIds: Tensor1D|TensorLike, numSegments: number): T {
  const $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
  const $segmentIds =
      convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
  assert(isInt(numSegments), () => 'numSegments must be of dtype int');

  const gradFunc = (dy: T, saved: Tensor[]) => {
    const [$segmentIds] = saved;
    const derX = () => {
      return gatherDropNegatives(dy, $segmentIds as Tensor1D);
    };
    return {$x: derX};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.unsortedSegmentSum($x, $segmentIds, numSegments);
    save([$segmentIds]);
    return res;
  }, {$x}, gradFunc) as T;
}

function gatherDropNegatives<T extends Tensor>(x: T, indices: Tensor1D) {
  // Helper function for unsorted segment ops. Gathers params for
  // positive segment ids and gathers 0 for inputs with negative segment id.
  // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
  const zeroClippedIndices = maximum(indices, zerosLike(indices));
  const gathered = gather(x, zeroClippedIndices as Tensor1D);
  let isPositive = greaterEqual(indices, scalar(0, 'int32'));
  const numIters = gathered.rank - isPositive.rank;
  for (let i = 0; i < numIters; ++i) {
    isPositive = expandDims(isPositive, i + 1);
  }
  isPositive = logicalAnd(isPositive, ones(gathered.shape, 'bool'));
  const zeroSlice = zerosLike(gathered);
  return where(isPositive, gathered, zeroSlice);
}

export const unsortedSegmentSum = op({unsortedSegmentSum_});
