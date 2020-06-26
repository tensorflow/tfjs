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
import {customGrad} from '../gradients';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as axis_util from './axis_util';
import {op} from './operation';
import {ones} from './tensor_ops';

/**
 * Computes the sum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If axes has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.sum().print();  // or tf.sum(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.sum(axis).print();  // or tf.sum(x, axis)
 * ```
 *
 * @param x The input tensor to compute the sum over. If the dtype is `bool`
 *   it will be converted to `int32` and the output dtype will be `int32`.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function sum_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  let $x = convertToTensor(x, 'x', 'sum');

  if ($x.dtype === 'bool') {
    $x = $x.toInt();
  }
  const axes = util.parseAxisParam(axis, $x.shape);

  // Use a custom gradient to bypass 2 gradient backprops since sum is used
  // extremely often.
  const customOp = customGrad((x: Tensor) => {
    const permutation = axis_util.getAxesPermutation(axes, x.rank);
    let reductionAxes = axes;
    let permutedX = x;
    if (permutation != null) {
      permutedX = x.transpose(permutation);
      reductionAxes = axis_util.getInnerMostAxes(reductionAxes.length, x.rank);
    }

    const gradFunc = (dy: Tensor) => {
      const expandedDyShape = x.shape.slice();
      axes.forEach(axis => {
        expandedDyShape[axis] = 1;
      });
      const expandedDy = dy.reshape(expandedDyShape);
      const derX = expandedDy.mul(ones(x.shape, 'float32'));
      return derX;
    };

    const gradInputs = (dy: Tensor) => {
      return {x: () => gradFunc(dy)};
    };

    const attrs = {axes: reductionAxes};
    let value = ENGINE.runKernelFunc(
        backend => backend.sum(permutedX, reductionAxes), {x: permutedX},
        gradInputs, 'Sum', attrs);

    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
      value = value.reshape(newShape);
    }

    return {value, gradFunc};
  });

  return customOp($x) as T;
}

export const sum = op({sum_});
