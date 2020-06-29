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
import {gradForMinAndMax} from './reduction_ops_util';
import {ones, scalar, zerosLike} from './tensor_ops';

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

/**
 * Computes the mean of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
 * true, the rank of the `tf.Tensor` is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and a `tf.Tensor` with
 * a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.mean().print();  // or tf.mean(a)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.mean(axis).print();  // or tf.mean(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function mean_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  const $x = convertToTensor(x, 'x', 'mean');

  const axes = util.parseAxisParam(axis, $x.shape);
  const shapes = axis_util.computeOutAndReduceShapes($x.shape, axes);
  const reduceShape = shapes[1];
  const reduceSize = util.sizeFromShape(reduceShape);

  // Use a custom gradient to bypass 2 gradient backprops since mean is used
  // extremely often.
  const customOp = customGrad((x: Tensor) => {
    const reduceSizeScalar = scalar(reduceSize);
    // Cast if needed.
    const xReduce =
        reduceSizeScalar.dtype === x.dtype ? x : x.cast(reduceSizeScalar.dtype);
    const res = xReduce.div(reduceSizeScalar);
    const value = res.sum(axis, keepDims);

    const gradFunc = (dy: Tensor) => {
      const expandedDyShape = x.shape.slice();
      axes.forEach(axis => {
        expandedDyShape[axis] = 1;
      });
      const expandedDy = dy.reshape(expandedDyShape);
      const derX = expandedDy.mul(ones(x.shape, 'float32')).div(reduceSize);
      return derX;
    };
    return {value, gradFunc};
  });

  return customOp($x) as T;
}

/**
 * Computes the minimum value from the input.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axes`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axes` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.min().print();  // or tf.min(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.min(axis).print();  // or tf.min(x, axis)
 * ```
 *
 * @param x The input Tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function min_<T extends Tensor>(
    x: Tensor|TensorLike, axis: number|number[] = null, keepDims = false): T {
  let $x = convertToTensor(x, 'x', 'min');
  const xOrig = $x;

  const origAxes = util.parseAxisParam(axis, $x.shape);
  let axes = origAxes;
  const permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
  if (permutedAxes != null) {
    $x = $x.transpose(permutedAxes);
    axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
  }

  const grad = (dy: T, saved: Tensor[]) =>
      gradForMinAndMax(dy, saved[1], saved[0], origAxes, permutedAxes);

  const inputsToSave = [$x];
  const outputsToSave: boolean[] = [true];
  let res = ENGINE.runKernelFunc((backend, save) => {
    const y = backend.min($x, axes);
    save([xOrig, y]);
    return y as T;
  }, {x: $x}, grad, 'Min', {axes}, inputsToSave, outputsToSave);
  if (keepDims) {
    const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
    res = res.reshape(newShape);
  }
  return res;
}

/**
 * Returns the indices of the minimum values along an `axis`.
 *
 * The result has the same shape as `input` with the dimension along `axis`
 * removed.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.argMin().print();  // or tf.argMin(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
 *
 * const axis = 1;
 * x.argMin(axis).print();  // or tf.argMin(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
 *
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function argMin_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  let $x = convertToTensor(x, 'x', 'argMin');

  if (axis == null) {
    axis = 0;
  }
  let axes = util.parseAxisParam(axis, $x.shape);
  const permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
  if (permutedAxes != null) {
    $x = $x.transpose(permutedAxes);
    axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
  }
  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {$x: () => zerosLike($x)};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.argMin($x, axes[0]);
    save([$x]);
    return res;
  }, {$x}, grad) as T;
}

/**
 * Returns the indices of the maximum values along an `axis`.
 *
 * The result has the same shape as `input` with the dimension along `axis`
 * removed.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.argMax().print();  // or tf.argMax(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
 *
 * const axis = 1;
 * x.argMax(axis).print();  // or tf.argMax(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
 */
/** @doc {heading: 'Operations', subheading: 'Reduction'} */
function argMax_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  let $x = convertToTensor(x, 'x', 'argMax');

  if (axis == null) {
    axis = 0;
  }
  let axes = util.parseAxisParam(axis, $x.shape);
  const permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
  if (permutedAxes != null) {
    $x = $x.transpose(permutedAxes);
    axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
  }
  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {x: () => zerosLike($x)};
  };
  const attrs = {axis: axes[0]};
  const inputsToSave = [$x];
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.argMax($x, axes[0]);
    save([$x]);
    return res;
  }, {x: $x}, grad, 'ArgMax', attrs, inputsToSave) as T;
}

export const argMax = op({argMax_});
export const argMin = op({argMin_});
export const mean = op({mean_});
export const min = op({min_});
export const sum = op({sum_});
