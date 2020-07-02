/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {parseAxisParam} from '../util';

import {getUndoAxesPermutation} from './axis_util';
import {op} from './operation';
import {collectGatherOpShapeInfo} from './segment_util';
import {unsortedSegmentSum} from './unsorted_segment_sum';

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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function gather_<T extends Tensor>(
    x: T|TensorLike, indices: Tensor|TensorLike, axis = 0): T {
  const $x = convertToTensor(x, 'x', 'gather');
  const $indices = convertToTensor(indices, 'indices', 'gather', 'int32');
  axis = parseAxisParam(axis, $x.shape)[0];
  const shapeInfo = collectGatherOpShapeInfo($x, $indices, axis);

  const grad = (dy: T, saved: Tensor[]) => {
    const [$indices] = saved;
    const derX = () => {
      const paramsShape = $x.shape;
      const indicesSize = $indices.size;

      const outerShape = paramsShape.slice(0, axis);
      const outerDims = outerShape.length;
      const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
      const innerDims = innerShape.length;

      const outerAxesIndices = arrayRange(0, outerDims);
      const innerAxesIndices =
          arrayRange(outerDims + 1, outerDims + 1 + innerDims);

      const valuesShape = arrayConcat([outerShape, [indicesSize], innerShape]);

      const values = dy.reshape(valuesShape);
      const reshapedIndices = $indices.reshape([indicesSize]);

      const transposeDims =
          arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
      const valuesTranspose = values.transpose(transposeDims);
      let paramsGrad = unsortedSegmentSum(
          valuesTranspose, reshapedIndices as Tensor1D, $x.shape[axis]);

      const invertTransposeDims = getUndoAxesPermutation(transposeDims);
      paramsGrad = paramsGrad.transpose(invertTransposeDims);

      return paramsGrad as T;
    };
    return {x: derX, indices: () => $indices};
  };
  return (ENGINE.runKernelFunc(
              (backend, save) => {
                const res = backend.gather($x, $indices.flatten(), axis);
                save([$indices]);
                return res;
              },
              {x: $x, indices: $indices}, grad, 'Gather', {axis}))
      .reshape(shapeInfo.outputShape);
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

export const gather = op({gather_});
