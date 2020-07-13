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
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';
import {scalar, zerosLike} from './tensor_ops';

/**
 * RReturns which elements of x are NaN.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isNaN().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isNaN_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'isNaN');

  // TODO(nsthorat): Let gradients be null for cases where we want to stop
  // backpropgation.
  const grad = (dy: T) => {
    return {$x: () => zerosLike(dy)};
  };
  return ENGINE.runKernelFunc(backend => backend.isNaN($x), {$x}, grad);
}

/**
 * Returns which elements of x are Infinity or -Infinity.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isInf().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isInf_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'isInf');

  // TODO(nsthorat): Let gradients be null for cases where we want to stop
  // backpropgation.
  const grad = (dy: T) => {
    return {$x: () => zerosLike(dy)};
  };
  return ENGINE.runKernelFunc(backend => backend.isInf($x), {$x}, grad);
}

/**
 * Returns which elements of x are finite.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isFinite().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function isFinite_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'isFinite');

  // TODO(nsthorat): Let gradients be null for cases where we want to stop
  // backpropgation.
  const grad = (dy: T) => {
    return {$x: () => zerosLike(dy)};
  };
  return ENGINE.runKernelFunc(backend => backend.isFinite($x), {$x}, grad);
}

/**
 * Computes round of input `tf.Tensor` element-wise: `round(x)`.
 * It implements banker's rounding.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.round().print();  // or tf.round(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function round_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'round');

  // TODO(nsthorat): Let gradients be null for cases where we want to stop
  // backpropgation.
  const grad = (dy: T) => {
    return {$x: () => zerosLike(dy)};
  };
  return ENGINE.runKernelFunc(backend => backend.round($x), {$x}, grad);
}

/**
 * Computes square root of the input `tf.Tensor` element-wise: `y = sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.sqrt().print();  // or tf.sqrt(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sqrt_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'sqrt');

  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {x: () => dy.div($x.toFloat().sqrt().mul(2))} as {x: () => T};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.sqrt($x);
    save([$x]);
    return res;
  }, {x: $x}, grad, 'Sqrt', {});
}

/**
 * Computes reciprocal of square root of the input `tf.Tensor` element-wise:
 * `y = 1 / sqrt(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 4, -1]);
 *
 * x.rsqrt().print();  // or tf.rsqrt(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function rsqrt_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'rsqrt');

  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {x: () => dy.div($x.pow(1.5).mul(2)).neg() as T};
  };
  const inputsToSave = [$x];
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.rsqrt($x);
    save([$x]);
    return res;
  }, {x: $x}, grad, 'Rsqrt', {} /* attrs */, inputsToSave);
}

/**
 * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
 *
 * ```js
 * const x = tf.tensor1d([0, -1, 2, -3]);
 *
 * x.sigmoid().print();  // or tf.sigmoid(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function sigmoid_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'sigmoid');

  const grad = (dy: T, saved: Tensor[]) => {
    const [y] = saved;
    return {x: () => dy.mul(y.mul(scalar(1).sub(y)))} as {x: () => T};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const y = backend.sigmoid($x);
    save([y]);
    return y;
  }, {x: $x}, grad, 'Sigmoid');
}

/**
 * Computes softplus of the input `tf.Tensor` element-wise: `log(exp(x) + 1)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.softplus().print();  // or tf.softplus(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function softplus_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'softplus');

  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {$x: () => dy.mul($x.sigmoid())} as {$x: () => T};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.softplus($x);
    save([$x]);
    return res;
  }, {$x}, grad);
}

/**
 * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha * x`
 *
 * ```js
 * const x = tf.tensor1d([0, 2, -1, -3]);
 *
 * x.step(.5).print();  // or tf.step(x, .5)
 * ```
 * @param x The input tensor.
 * @param alpha The gradient when input is negative.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function step_<T extends Tensor>(x: T|TensorLike, alpha = 0.0): T {
  const $x = convertToTensor(x, 'x', 'step');

  // TODO(manrajgrover): Return null for gradients when backprop supports
  // it.
  const grad = (dy: T) => {
    return {$x: () => zerosLike(dy)};
  };
  return ENGINE.runKernelFunc(backend => backend.step($x, alpha), {$x}, grad);
}

export const round = op({round_});
export const rsqrt = op({rsqrt_});
export const sigmoid = op({sigmoid_});
export const isNaN = op({isNaN_});
export const isInf = op({isInf_});
export const isFinite = op({isFinite_});
export const softplus = op({softplus_});
export const sqrt = op({sqrt_});
export const step = op({step_});
