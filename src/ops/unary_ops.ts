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
import {Tensor} from '../tensor';
import * as util from '../util';
import {operation} from './operation';
import * as ops from './ops';
import {zerosLike} from './ops';
import * as selu_util from './selu_util';

export class UnaryOps {
  /**
   * Computes `-1 * x` element-wise.
   *
   * ```js
   * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
   *
   * x.neg().print();  // or tf.neg(x)
   * ```
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static neg<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'neg');

    const grad = (dy: T) => {
      return {x: () => dy.neg()};
    };
    return ENV.engine.runKernel(backend => backend.neg(x), {x}, grad);
  }

  /**
   * Computes ceiling of input `Tensor` element-wise: `ceil(x)`
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.ceil().print();  // or tf.ceil(x)
   * ```
   * @param x The input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static ceil<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'ceil');

    // TODO(manrajgrover): Return null for gradients when backprop supports it.
    const grad = (dy: T) => {
      return {x: () => ops.zerosLike(dy)};
    };
    return ENV.engine.runKernel(backend => backend.ceil(x), {x}, grad);
  }

  /**
   * Computes floor of input `Tensor` element-wise: `floor(x)`.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.floor().print();  // or tf.floor(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static floor<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'floor');

    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy: T) => {
      return {x: () => ops.zerosLike(dy)};
    };
    return ENV.engine.runKernel(backend => backend.floor(x), {x}, grad);
  }

  /**
   * Returns an element-wise indication of the sign of a number.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
   *
   * x.sign().print();  // or tf.sign(x)
   * ```
   * @param x The input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sign<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'sign');

    const grad = (dy: T) => {
      return {x: () => ops.zerosLike(dy)};
    };
    return ENV.engine.runKernel(backend => backend.sign(x), {x}, grad);
  }

  /**
   * Computes round of input `Tensor` element-wise: `round(x)`.
   * It implements banker's rounding.
   *
   * ```js
   * const x = tf.tensor1d([.6, 1.1, -3.3]);
   *
   * x.round().print();  // or tf.round(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static round<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'round');

    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const grad = (dy: T) => {
      return {x: () => ops.zerosLike(dy)};
    };
    return ENV.engine.runKernel(backend => backend.round(x), {x}, grad);
  }

  /**
   * Computes exponential of the input `Tensor` element-wise. `e ^ x`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, -3]);
   *
   * x.exp().print();  // or tf.exp(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static exp<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'exp');

    const bck = (dy: T, saved: Tensor[]) => {
      const [y] = saved;
      return {x: () => dy.mulStrict(y as T)};
    };
    return ENV.engine.runKernel(
        (backend, save) => save(backend.exp(x)), {x}, bck);
  }

  /**
   * Computes exponential of the input `Tensor` minus one element-wise.
   * `e ^ x - 1`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, -3]);
   *
   * x.expm1().print();  // or tf.expm1(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static expm1<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'expm1');

    const grad = (dy: T) => {
      return {x: () => dy.mulStrict(x.exp())};
    };
    return ENV.engine.runKernel(backend => backend.expm1(x), {x}, grad);
  }

  /**
   * Computes natural logarithm of the input `Tensor` element-wise: `ln(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.E]);
   *
   * x.log().print();  // or tf.log(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static log<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'log');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(x.toFloat())};
    };
    return ENV.engine.runKernel(backend => backend.log(x), {x}, grad);
  }

  /**
   * Computes natural logarithm of the input `Tensor` plus one
   * element-wise: `ln(1 + x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.E - 1]);
   *
   * x.log1p().print();  // or tf.log1p(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static log1p<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'log1p');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(x.add(ops.scalar(1)))};
    };
    return ENV.engine.runKernel(backend => backend.log1p(x), {x}, grad);
  }

  /**
   * Computes square root of the input `Tensor` element-wise: `y = sqrt(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 4, -1]);
   *
   * x.sqrt().print();  // or tf.sqrt(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sqrt<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'sqrt');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(x.toFloat().sqrt().mul(ops.scalar(2)))};
    };
    return ENV.engine.runKernel(backend => backend.sqrt(x), {x}, grad);
  }

  /**
   * Computes reciprocal of square root of the input `Tensor` element-wise:
   * `y = 1 / sqrt(x)`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 4, -1]);
   *
   * x.rsqrt().print();  // or tf.rsqrt(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static rsqrt<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'rsqrt');

    const grad = (dy: T) => {
      return {
        x: () => dy.divStrict(x.pow(ops.scalar(1.5)).mul(ops.scalar(2))).neg()
      };
    };
    return ENV.engine.runKernel(backend => backend.rsqrt(x), {x}, grad);
  }

  /**
   * Computes square of `x` element-wise: `x ^ 2`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);
   *
   * x.square().print();  // or tf.square(x)
   * ```
   * @param x The input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static square<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'square');

    const grad = (dy: T) => {
      return {x: () => dy.mulStrict(x.toFloat().mul(ops.scalar(2)))};
    };
    return ENV.engine.runKernel(backend => backend.square(x), {x}, grad);
  }

  /**
   * Computes reciprocal of x element-wise: `1 / x`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, 2]);
   *
   * x.reciprocal().print();  // or tf.reciprocal(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static reciprocal<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'reciprocal');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(x.square().neg())};
    };
    return ENV.engine.runKernel(backend => backend.reciprocal(x), {x}, grad);
  }

  /**
   * Computes absolute value element-wise: `abs(x)`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.abs().print();  // or tf.abs(x)
   * ```
   * @param x The input `Tensor`.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static abs<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'abs');

    const grad = (dy: T) => {
      return {x: () => dy.mulStrict(x.toFloat().step(-1))};
    };
    return ENV.engine.runKernel(backend => backend.abs(x), {x}, grad);
  }

  /**
   * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
   * ```
   * @param x The input tensor.
   * @param clipValueMin Lower-bound of range to be clipped to.
   * @param clipValueMax Upper-bound of range to be clipped to.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static clipByValue<T extends Tensor>(
      x: T, clipValueMin: number, clipValueMax: number): T {
    util.assertArgumentsAreTensors({x}, 'clipByValue');
    util.assert(
        (clipValueMin <= clipValueMax),
        `Error in clip: min (${clipValueMin}) must be ` +
            `less than or equal to max (${clipValueMax}).`);

    const grad = (dy: T) => {
      return {
        x: () => dy.where(
                     x.greaterEqual(ops.scalar(clipValueMin))
                         .logicalAnd(x.lessEqual(ops.scalar(clipValueMax))),
                     zerosLike(dy)) as T,
      };
    };
    return ENV.engine.runKernel(
        backend => backend.clip(x, clipValueMin, clipValueMax), {x}, grad);
  }

  /**
   * Computes rectified linear element-wise: `max(x, 0)`
   *
   * ```js
   * const x = tf.tensor1d([-1, 2, -3, 4]);
   *
   * x.relu().print();  // or tf.relu(x)
   * ```
   * @param x The input tensor. If the dtype is `bool`, the output dtype will be
   *     `int32'.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static relu<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'relu');

    if (x.dtype === 'bool') {
      return x.toInt();
    }
    const grad = (dy: T) => {
      const stepRes = x.step();
      return {x: () => dy.mulStrict(stepRes.toFloat())};
    };
    return ENV.engine.runKernel(backend => backend.relu(x), {x}, grad);
  }

  /**
   * Computes exponential linear element-wise, `x > 0 ? e ^ x - 1 : 0`
   *
   * ```js
   * const x = tf.tensor1d([-1, 1, -3, 2]);
   *
   * x.elu().print();  // or tf.elu(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static elu<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'elu');

    const grad = (dy: T, saved: Tensor[]) => {
      const [y] = saved;
      return {
        x: () =>
            ENV.engine.runKernel(backend => backend.eluDer(dy, y), {dy, y}) as T
      };
    };
    return ENV.engine.runKernel(
        (backend, save) => save(backend.elu(x)), {x}, grad);
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
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static selu<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'selu');

    const grad = (dy: T) => {
      return {
        x: () => {
          const mask = x.greater(ops.scalar(0));

          const scaleAlpha = ops.scalar(selu_util.SELU_SCALEALPHA);
          const scale = ops.scalar(selu_util.SELU_SCALE);

          const greaterThanZeroDer = dy.mul(scale);
          const lessEqualZeroDer = dy.mul(scaleAlpha).mul(x.toFloat().exp());

          return ops.where(mask, greaterThanZeroDer, lessEqualZeroDer) as T;
        }
      };
    };
    return ENV.engine.runKernel(backend => backend.selu(x), {x}, grad);
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
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static leakyRelu<T extends Tensor>(x: T, alpha = 0.2): T {
    util.assertArgumentsAreTensors({x}, 'leakyRelu');

    return ops.maximum(ops.scalar(alpha).mul(x), x);
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
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static prelu<T extends Tensor>(x: T, alpha: T): T {
    util.assertArgumentsAreTensors({x, alpha}, 'prelu');

    const zero = ops.scalar(0);
    return ops.maximum(zero, x).add(alpha.mul(ops.minimum(zero, x)));
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
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sigmoid<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'sigmoid');

    const grad = (dy: T, saved: Tensor[]) => {
      const [y] = saved;
      return {x: () => dy.mulStrict(y.mul(ops.scalar(1).sub(y)))};
    };
    return ENV.engine.runKernel(
        (backend, save) => save(backend.sigmoid(x)), {x}, grad);
  }

  /**
   * Computes log sigmoid of the input `Tensor` element-wise:
   * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.logSigmoid().print();  // or tf.logSigmoid(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static logSigmoid<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'logSigmoid');

    const grad = (dy: T) => {
      return {x: () => dy.mulStrict(x.neg().sigmoid())};
    };
    return ENV.engine.runKernel(
        backend => backend.softplus(x.neg()).neg(), {x}, grad);
  }

  /**
   * Computes softplus of the input `Tensor` element-wise: `log(exp(x) + 1)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.softplus().print();  // or tf.softplus(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static softplus<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'softplus');

    const grad = (dy: T) => {
      return {x: () => dy.mulStrict(x.sigmoid())};
    };
    return ENV.engine.runKernel(backend => backend.softplus(x), {x}, grad);
  }

  /**
   * Computes sin of the input Tensor element-wise: `sin(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.sin().print();  // or tf.sin(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sin<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'sin');

    const grad = (dy: T) => {
      return {x: () => x.toFloat().cos().mulStrict(dy)};
    };
    return ENV.engine.runKernel(backend => backend.sin(x), {x}, grad);
  }

  /**
   * Computes cos of the input `Tensor` element-wise: `cos(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.cos().print();  // or tf.cos(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static cos<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'cos');

    const grad = (dy: T) => {
      return {x: () => x.toFloat().sin().neg().mulStrict(dy)};
    };
    return ENV.engine.runKernel(backend => backend.cos(x), {x}, grad);
  }

  /**
   * Computes tan of the input `Tensor` element-wise, `tan(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
   *
   * x.tan().print();  // or tf.tan(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static tan<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'tan');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(x.cos().square())};
    };
    return ENV.engine.runKernel(backend => backend.tan(x), {x}, grad);
  }

  /**
   * Computes asin of the input `Tensor` element-wise: `asin(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.asin().print();  // or tf.asin(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static asin<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'asin');

    const grad = (dy: T) => {
      return {
        x: () =>
            dy.divStrict(UnaryOps.sqrt(ops.scalar(1).sub(x.toFloat().square())))
      };
    };
    return ENV.engine.runKernel(backend => backend.asin(x), {x}, grad);
  }

  /**
   * Computes acos of the input `Tensor` element-wise: `acos(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.acos().print();  // or tf.acos(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static acos<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'acos');

    const grad = (dy: T) => {
      return {
        x: () =>
            dy.divStrict(UnaryOps.sqrt(ops.scalar(1).sub(x.toFloat().square())))
                .neg()
      };
    };
    return ENV.engine.runKernel(backend => backend.acos(x), {x}, grad);
  }

  /**
   * Computes atan of the input `Tensor` element-wise: `atan(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.atan().print();  // or tf.atan(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static atan<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'atan');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(ops.scalar(1).add(x.toFloat().square()))};
    };
    return ENV.engine.runKernel(backend => backend.atan(x), {x}, grad);
  }

  /**
   * Computes hyperbolic sin of the input `Tensor` element-wise: `sinh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.sinh().print();  // or tf.sinh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sinh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'sinh');

    const grad = (dy: T) => {
      return {x: () => x.toFloat().cosh().mulStrict(dy)};
    };
    return ENV.engine.runKernel(backend => backend.sinh(x), {x}, grad);
  }

  /**
   * Computes hyperbolic cos of the input `Tensor` element-wise: `cosh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.cosh().print();  // or tf.cosh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static cosh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'cosh');

    const grad = (dy: T) => {
      return {x: () => x.toFloat().sinh().mulStrict(dy)};
    };
    return ENV.engine.runKernel(backend => backend.cosh(x), {x}, grad);
  }

  /**
   * Computes hyperbolic tangent of the input `Tensor` element-wise: `tanh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, 70]);
   *
   * x.tanh().print();  // or tf.tanh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static tanh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'tanh');

    const grad = (dy: T, saved: Tensor[]) => {
      const [y] = saved;
      return {x: () => ops.scalar(1).sub(y.square()).mulStrict(dy) as T};
    };
    return ENV.engine.runKernel(
        (backend, save) => save(backend.tanh(x)), {x}, grad);
  }

  /**
   * Computes inverse hyperbolic sin of the input `Tensor` element-wise:
   * `asinh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, 1, -1, .7]);
   *
   * x.asinh().print();  // or tf.asinh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static asinh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'asinh');

    const grad = (dy: T) => {
      return {
        x: () =>
            dy.divStrict(UnaryOps.sqrt(ops.scalar(1).add(x.toFloat().square())))
      };
    };
    return ENV.engine.runKernel(backend => backend.asinh(x), {x}, grad);
  }

  /**
   * Computes the inverse hyperbolic cos of the input `Tensor` element-wise:
   * `acosh(x)`
   *
   * ```js
   * const x = tf.tensor1d([10, 1, 3, 5.7]);
   *
   * x.acosh().print();  // or tf.acosh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static acosh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'acosh');

    const grad = (dy: T) => {
      return {
        x: () =>
            dy.divStrict(UnaryOps.sqrt(x.toFloat().square().sub(ops.scalar(1))))
      };
    };
    return ENV.engine.runKernel(backend => backend.acosh(x), {x}, grad);
  }

  /**
   * Computes inverse hyperbolic tan of the input `Tensor` element-wise:
   * `atanh(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, .1, -.1, .7]);
   *
   * x.atanh().print();  // or tf.atanh(x)
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static atanh<T extends Tensor>(x: T): T {
    util.assertArgumentsAreTensors({x}, 'atanh');

    const grad = (dy: T) => {
      return {x: () => dy.divStrict(ops.scalar(1).sub(x.toFloat().square()))};
    };
    return ENV.engine.runKernel(backend => backend.atanh(x), {x}, grad);
  }

  /**
   * Computes gause error function of the input `Tensor` element-wise:
   * `erf(x)`
   *
   * ```js
   * const x = tf.tensor1d([0, .1, -.1, .7]);
   *
   * x.erf().print(); // or tf.erf(x);
   * ```
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static erf<T extends Tensor>(x: T): T {
    util.assert(x.dtype === 'int32' || x.dtype === 'float32',
        'Input dtype must be `int32` or `float32`.');

    if (x.dtype === 'int32') {
      x = x.toFloat();
    }

    const grad = (dy: T) => {
      return {
        x: () =>
            dy.mulStrict(ops.scalar(2/Math.sqrt(Math.PI))
                .mul(x.square().neg().exp()))
      };
    };
    return ENV.engine.runKernel(backend => backend.erf(x), {x}, grad);
  }

  /**
   * Computes step of the input `Tensor` element-wise: `x > 0 ? 1 : alpha * x`
   *
   * ```js
   * const x = tf.tensor1d([0, 2, -1, -3]);
   *
   * x.step(.5).print();  // or tf.step(x, .5)
   * ```
   * @param x The input tensor.
   * @param alpha The gradient when input is negative.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static step<T extends Tensor>(x: T, alpha = 0.0): T {
    util.assertArgumentsAreTensors({x}, 'step');

    // TODO(manrajgrover): Return null for gradients when backprop supports it.
    const grad = (dy: T) => {
      return {x: () => ops.zerosLike(dy)};
    };
    return ENV.engine.runKernel(backend => backend.step(x, alpha), {x}, grad);
  }
}
