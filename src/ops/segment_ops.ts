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
import {assertArgumentsAreTensors} from '../tensor_util';
import * as util from '../util';
import {ArrayOps} from './array_ops';
import {getUndoAxesPermutation, parseAxisParam} from './axis_util';
import {BinaryOps} from './binary_ops';
import {CompareOps} from './compare';
import {LogicalOps} from './logical_ops';
import {operation} from './operation';
import {TensorOps} from './tensor_ops';

export class SegmentOps {
  /**
   * Computes the sum along segments of a `Tensor`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
   * const numSegments = 3;
   *
   * x.unsortedSegmentSum(segmentIds, numSegments).print()
   * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
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
    assertArgumentsAreTensors({x, segmentIds}, 'unsortedSegmentSum');
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

  /**
   * Gather slices from tensor `x`'s axis `axis` according to `indices`.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   * const indices = tf.tensor1d([1, 3, 3], 'int32');
   *
   * x.gather(indices).print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   * const indices = tf.tensor1d([1, 1, 0], 'int32');
   *
   * x.gather(indices).print();
   * ```
   * @param x The input tensor whose slices to be gathered.
   * @param indices The indices of the values to extract.
   * @param axis The axis over which to select values. Defaults to 0.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static gather<T extends Tensor>(x: T, indices: Tensor1D, axis = 0): T {
    assertArgumentsAreTensors({x, indices}, 'gather');

    util.assert(indices.dtype === 'int32', 'Indices must be of dtype `int32`');
    axis = parseAxisParam(axis, x.shape)[0];
    const grad = (dy: T) => {
      const derX = () => {
        if (axis === 0) {
          return SegmentOps.unsortedSegmentSum(dy, indices, x.shape[axis]);
        }
        const paramsShape = x.shape;
        const indicesSize = indices.size;

        const outerShape = paramsShape.slice(0, axis);
        const outerDims = outerShape.length;
        const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
        const innerDims = innerShape.length;

        const outerAxesIndices = arrayRange(0, outerDims);
        const innerAxesIndices =
            arrayRange(outerDims + 1, outerDims + 1 + innerDims);

        const valuesShape =
            arrayConcat([outerShape, [indicesSize], innerShape]);

        const values = dy.reshape(valuesShape);
        const reshapedIndices = indices.reshape([indicesSize]);

        const transposeDims =
            arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
        const valuesTranspose = values.transpose(transposeDims);

        let paramsGrad = SegmentOps.unsortedSegmentSum(
            valuesTranspose, reshapedIndices as Tensor1D, x.shape[axis]);

        const invertTransposeDims = getUndoAxesPermutation(transposeDims);
        paramsGrad = paramsGrad.transpose(invertTransposeDims);

        return paramsGrad as T;
      };
      return {x: derX};
    };
    return ENV.engine.runKernel(
        backend => backend.gather(x, indices, axis), {x}, grad);
  }
}

function arrayRange(start: number, stop: number): number[] {
  const result = [];
  for (let i = start; i < stop; ++i) {
    result.push(i);
  }
  return result;
}

function arrayConcat(arrays: number[][]): number[] {
  const result = [];
  for (let i = 0; i < arrays.length; ++i) {
    for (let j = 0; j < arrays[i].length; ++j) {
      result.push(arrays[i][j]);
    }
  }
  return result;
}

function gatherDropNegatives<T extends Tensor>(x: T, indices: Tensor1D) {
  // Helper function for unsorted segment ops. Gathers params for
  // positive segment ids and gathers 0 for inputs with negative segment id.
  // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
  const zeroClippedIndices =
      BinaryOps.maximum(indices, TensorOps.zerosLike(indices));
  const gathered = SegmentOps.gather(x, zeroClippedIndices as Tensor1D);
  let isPositive =
      CompareOps.greaterEqual(indices, TensorOps.scalar(0, 'int32'));
  const numIters = gathered.rank - isPositive.rank;
  for (let i = 0; i < numIters; ++i) {
    isPositive = ArrayOps.expandDims(isPositive, i + 1);
  }
  isPositive =
      LogicalOps.logicalAnd(isPositive, TensorOps.ones(gathered.shape, 'bool'));
  const zeroSlice = TensorOps.zerosLike(gathered);
  return LogicalOps.where(isPositive, gathered, zeroSlice);
}
