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
import {maximum} from './binary_ops';
import {getReductionAxes} from './broadcast_util';
import {where} from './logical_ops';
import {op} from './operation';
import {SELU_SCALE, SELU_SCALEALPHA} from './selu_util';
import {scalar, zerosLike} from './tensor_ops';

/**
 * Computes rectified linear element-wise: `max(x, 0)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.relu().print();  // or tf.relu(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function relu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'relu');

  if ($x.dtype === 'bool') {
    return $x.toInt();
  }
  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {$x: () => dy.mulStrict($x.step().toFloat() as T)};
  };
  return ENGINE.runKernel((backend, save) => {
    const res = backend.relu($x);
    save([$x]);
    return res;
  }, {$x}, grad);
}

/**
 * Computes exponential linear element-wise: `x > 0 ? e ^ x - 1 : 0`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 1, -3, 2]);
 *
 * x.elu().print();  // or tf.elu(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function elu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'elu');

  const grad = (dy: T, saved: Tensor[]) => {
    const [y] = saved;
    return {
      $x: () => ENGINE.runKernel(backend => backend.eluDer(dy, y), {dy, y}) as T
    };
  };
  return ENGINE.runKernel((backend, save) => {
    const y = backend.elu($x);
    save([y]);
    return y;
  }, {$x}, grad);
}

/**
 * Computes scaled exponential linear element-wise.
 *
 * `x < 0 ? scale * alpha * (exp(x) - 1) : x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.selu().print();  // or tf.selu(x)
 * ```
 * @param x The input tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function selu_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'selu');

  const grad = (dy: T, saved: Tensor[]) => {
    const [$x] = saved;
    return {
      $x: () => {
        const mask = $x.greater(scalar(0));

        const scaleAlpha = scalar(SELU_SCALEALPHA);
        const scale = scalar(SELU_SCALE);

        const greaterThanZeroDer = dy.mul(scale);
        const lessEqualZeroDer = dy.mul(scaleAlpha).mul($x.toFloat().exp());

        return where(mask, greaterThanZeroDer, lessEqualZeroDer) as T;
      }
    };
  };
  return ENGINE.runKernel((backend, save) => {
    const res = backend.selu($x);
    save([$x]);
    return res;
  }, {$x}, grad);
}

/**
 * Computes leaky rectified linear element-wise.
 *
 * See
 * [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
 *     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.leakyRelu(0.1).print();  // or tf.leakyRelu(x, 0.1)
 * ```
 * @param x The input tensor.
 * @param alpha The scaling factor for negative values, defaults to 0.2.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function leakyRelu_<T extends Tensor>(x: T|TensorLike, alpha = 0.2): T {
  const $x = convertToTensor(x, 'x', 'leakyRelu');
  return maximum(scalar(alpha).mul($x), $x);
}

/**
 * Computes leaky rectified linear element-wise with parametric alphas.
 *
 * `x < 0 ? alpha * x : f(x) = x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 * const alpha = tf.scalar(0.1);
 *
 * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
 * ```
 * @param x The input tensor.
 * @param alpha Scaling factor for negative values.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function prelu_<T extends Tensor>(x: T|TensorLike, alpha: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'prelu');
  const $alpha = convertToTensor(alpha, 'alpha', 'prelu');

  const grad = (dy: Tensor, saved: Tensor[]) => {
    const [$x, $alpha] = saved;
    const mask = $x.greater(0);

    return {
      $x: () => where(mask, dy, dy.mul($alpha)) as T,
      $alpha: () => {
        let res = where(mask, zerosLike(dy), dy.mul($x));
        const reduceAxes = getReductionAxes($alpha.shape, dy.shape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape($alpha.shape) as T;
      }
    };
  };

  return ENGINE.runKernel((backend, save) => {
    const res = backend.prelu($x, $alpha);
    save([$x, $alpha]);
    return res;
  }, {$x, $alpha}, grad) as T;
}

export const elu = op({elu_});
export const leakyRelu = op({leakyRelu_});
export const prelu = op({prelu_});
export const relu = op({relu_});
export const selu = op({selu_});
