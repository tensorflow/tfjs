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

import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor, Tensor1D} from '../tensor';
import * as util from '../util';
import {ArrayOps} from './array_ops';
import {BinaryOps} from './binary_ops';
import {CompareOps} from './compare';
import {LogicalOps} from './logical_ops';
import {operation} from './operation';

export class SegmentOps {
  /**
   * Computes the sum along segments of a `Tensor`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
   * comst numSegments = 3;
   *
   * x.unsortedSegmentSum(indices, numSegments).print()
   * //or tf.unsortedSegmentSum(x, indices, numSegments)
   * ```
   * @param x The `Tensor` that will be summed along its segments
   * @param segmentIds A `Tensor1D` whose rank is equal to the rank of `x`'s
   * dimension along the `axis`.  Maps each element of `x` to a segment.
   * @param numSegments The number of distinct `segmentIds`
   */
  @doc({heading: 'Operations', subheading: 'Segment'})
  @operation
  static unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): T {
    util.assertArgumentsAreTensors({x, segmentIds}, 'unsortedSegmentSum');
    util.assert(
        segmentIds.dtype === 'int32', 'segmentIds must be of dtype `int32`');
    util.assert(util.isInt(numSegments), 'numSegments must be of dtype int');

    const gradFunc = (dy: T) => {
      const derX = () => {
        return gatherDropNegatives(dy, segmentIds);
      };
      return {x: derX};
    };
    return ENV.engine.runKernel(
               backend =>
                   backend.unsortedSegmentSum(x, segmentIds, numSegments),
               {x}, gradFunc) as T;
  }
}

function gatherDropNegatives<T extends Tensor>(x: T, indices: Tensor1D) {
  // Helper function for unsorted segment ops. Gathers params for
  // positive segment ids and gathers 0 for inputs with negative segment id.
  // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
  const zeroClippedIndices =
      BinaryOps.maximum(indices, ArrayOps.zerosLike(indices));
  const gathered = ArrayOps.gather(x, zeroClippedIndices as Tensor1D);
  let isPositive =
      CompareOps.greaterEqual(indices, ArrayOps.scalar(0, 'int32'));
  const numIters = gathered.rank - isPositive.rank;
  for (let i = 0; i < numIters; ++i) {
    isPositive = ArrayOps.expandDims(isPositive, i + 1);
  }
  isPositive =
      LogicalOps.logicalAnd(isPositive, ArrayOps.ones(gathered.shape, 'bool'));
  const zeroSlice = ArrayOps.zerosLike(gathered);
  return LogicalOps.where(isPositive, gathered, zeroSlice);
}
