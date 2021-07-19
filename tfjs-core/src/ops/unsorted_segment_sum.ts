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

import {ENGINE} from '../engine';
import {UnsortedSegmentSum, UnsortedSegmentSumAttrs, UnsortedSegmentSumInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor1D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, isInt} from '../util';

import {op} from './operation';

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
 *
 * @doc {heading: 'Operations', subheading: 'Segment'}
 */
function unsortedSegmentSum_<T extends Tensor>(
    x: T|TensorLike, segmentIds: Tensor1D|TensorLike, numSegments: number): T {
  const $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
  const $segmentIds =
      convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
  assert(isInt(numSegments), () => 'numSegments must be of dtype int');

  const inputs: UnsortedSegmentSumInputs = {x: $x, segmentIds: $segmentIds};
  const attrs: UnsortedSegmentSumAttrs = {numSegments};

  return ENGINE.runKernel(
      UnsortedSegmentSum, inputs as {} as NamedTensorMap,
      attrs as {} as NamedAttrMap);
}

export const unsortedSegmentSum = op({unsortedSegmentSum_});
