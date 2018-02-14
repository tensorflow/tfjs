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

export class Ops {
  /**
   * Computes `-1 * x` element-wise.
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static neg<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Neg', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.neg()};
    }) as T;
  }

  /**
   * Computes ceiling of input `Tensor` element-wise: `ceil(x)`
   *
   * @param x The input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static ceil<T extends Tensor>(x: T): T {
    // TODO(manrajgrover): Return null for gradients when backprop supports it.
    const gradient = (dy: T, y: T) => {
      return {x: () => ops.zeros(y.shape)};
    };
    return ENV.engine.executeKernel('Ceil', {inputs: {x}}, gradient) as T;
  }

  /**
   * Computes floor of input `Tensor` element-wise: `floor(x)`.
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static floor<T extends Tensor>(x: T): T {
    // TODO(nsthorat): Let gradients be null for cases where we want to stop
    // backpropgation.
    const gradient = (dy: T, y: T) => {
      return {x: () => ops.zeros(y.shape)};
    };
    return ENV.engine.executeKernel('Floor', {inputs: {x}}, gradient) as T;
  }

  /**
   * Computes exponential of the input `Tensor` element-wise. `e ^ x`
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static exp<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Exp', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.mul(y)};
    }) as T;
  }

  /**
   * Computes natural logarithm of the input `Tensor` element-wise: `ln(x)`
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static log<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Log', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.div(x.toFloat())};
    }) as T;
  }

  /**
   * Computes square root of the input `Tensor` element-wise: `y = sqrt(x)`
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sqrt<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Sqrt', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.div(x.toFloat().sqrt().mul(ops.scalar(2)))};
    }) as T;
  }

  /**
   * Computes square of `x` element-wise: `x ^ 2`
   *
   * @param x The input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static square<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Square', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.mul(x.toFloat().mul(ops.scalar(2)))};
    }) as T;
  }

  /**
   * Computes absolute value element-wise: `abs(x)`
   *
   * @param x The input `Tensor`.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static abs<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Abs', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.mul(x.toFloat().step(-1))};
    }) as T;
  }

  /**
   * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
   *
   * @param x The input tensor.
   * @param clipValueMin Lower-bound of range to be clipped to.
   * @param clipValueMax Upper-bound of range to be clipped to.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static clipByValue<T extends Tensor>(
      x: T, clipValueMin: number, clipValueMax: number): T {
    util.assert(
        (clipValueMin <= clipValueMax),
        `Error in clip: min (${clipValueMin}) must be` +
            `less than or equal to max (${clipValueMax}).`);
    return ENV.engine.executeKernel(
               'Clip',
               {inputs: {x}, args: {min: clipValueMin, max: clipValueMax}},
               (dy: T, y: T) => {
                 return {
                   // TODO(cais): Fix gradients for the case where x = min or x
                   // = max.
                   x: () => dy.where(
                       x.greater(ops.scalar(clipValueMin))
                           .logicalAnd(x.less(ops.scalar(clipValueMax))),
                       zerosLike(dy)),
                 };
               }) as T;
  }

  /**
   * Computes rectified linear element-wise: `max(x, 0)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static relu<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Relu', {inputs: {x}}, (dy: T, y: T) => {
      const stepRes = x.step() as Tensor;
      return {x: () => dy.mul(stepRes.toFloat())};
    }) as T;
  }

  /**
   * Computes exponential linear element-wise, `x > 0 ? e ^ x - 1 : 0`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static elu<T extends Tensor>(x: T): T {
    const der = (dy: Tensor) => {
      return {
        x: () => dy.mul(eluDer(x)),
        alpha: () => {
          throw new Error(
              'Derivative of prelu with respect to alpha is ' +
              'not implemented yet');
        }
      };
    };
    return ENV.engine.executeKernel('Elu', {inputs: {x}}, der) as T;
  }

  /**
   * Computes scaled exponential linear element-wise.
   *
   * `x < 0 ? scale * alpha * (exp(x) - 1) : x`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static selu<T extends Tensor>(x: T): T {
    const gradient = (dy: T, y: T) => {
      return {
        x: () => {
          const mask = x.greater(ops.scalar(0));

          const scaleAlpha = ops.scalar(selu_util.SELU_SCALEALPHA);
          const scale = ops.scalar(selu_util.SELU_SCALE);

          const greaterThanZeroDer = dy.mul(scale);
          const lessEqualZeroDer = dy.mul(scaleAlpha).mul(x.toFloat().exp());

          const res = ops.where(mask, greaterThanZeroDer, lessEqualZeroDer);

          return res;
        }
      };
    };
    return ENV.engine.executeKernel('Selu', {inputs: {x}}, gradient) as T;
  }

  /**
   * Computes leaky rectified linear element-wise.
   *
   * See
   * [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
   *     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
   *
   * @param x The input tensor.
   * @param alpha The scaling factor for negative values, defaults to 0.2.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static leakyRelu<T extends Tensor>(x: T, alpha = 0.2): T {
    const gradient = (dy: T, y: T) => {
      return {x: () => dy.mul(x.step(alpha))};
    };
    return ENV.engine.executeKernel(
               'LeakyRelu', {inputs: {x}, args: {alpha}}, gradient) as T;
  }

  /**
   * Computes leaky rectified linear element-wise with parametric alphas.
   *
   * `x < 0 ? alpha * x : f(x) = x`
   *
   * @param x The input tensor.
   * @param alpha Scaling factor for negative values.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static prelu<T extends Tensor>(x: T, alpha: T): T {
    const der = (dy: Tensor) => {
      return {
        x: () => dy.mul(preluDer(x, alpha)),
        alpha: () => {
          throw new Error(
              'Derivative of prelu with respect to alpha is ' +
              'not implemented yet');
        }
      };
    };
    return ENV.engine.executeKernel('PReLU', {inputs: {x, alpha}}, der) as T;
  }

  /**
   * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sigmoid<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Sigmoid', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.mul(y.mul(ops.scalar(1).sub(y)))};
    }) as T;
  }

  /**
   * Computes sin of the input Tensor element-wise: `sin(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sin<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Sin', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => x.toFloat().cos().mul(dy)};
    }) as T;
  }

  /**
   * Computes cos of the input `Tensor` element-wise: `cos(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static cos<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Cos', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => x.toFloat().sin().neg().mul(dy)};
    }) as T;
  }

  /**
   * Computes tan of the input `Tensor` element-wise, `tan(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static tan<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Tan', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.div(x.cos().square())};
    }) as T;
  }

  /**
   * Computes asin of the input `Tensor` element-wise: `asin(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static asin<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Asin', {inputs: {x}}, (dy: T, y: T) => {
      return {
        x: () => dy.div(Ops.sqrt(ops.scalar(1).sub(x.toFloat().square())))
      };
    }) as T;
  }

  /**
   * Computes acos of the input `Tensor` element-wise: `acos(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static acos<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Acos', {inputs: {x}}, (dy: T, y: T) => {
      return {
        x: () => dy.div(Ops.sqrt(ops.scalar(1).sub(x.toFloat().square()))).neg()
      };
    }) as T;
  }

  /**
   * Computes atan of the input `Tensor` element-wise: `atan(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static atan<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Atan', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => dy.div(ops.scalar(1).add(x.toFloat().square()))};
    }) as T;
  }

  /**
   * Computes hyperbolic sin of the input `Tensor` element-wise: `sinh(x)`
   *
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static sinh<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Sinh', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => x.toFloat().cosh().mul(dy)};
    }) as T;
  }

  /**
   * Computes hyperbolic cos of the input `Tensor` element-wise: `cosh(x)`
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static cosh<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Cosh', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => x.toFloat().sinh().mul(dy)};
    }) as T;
  }

  /**
   * Computes hyperbolic tangent of the input `Tensor` element-wise: `tanh(x)`
   * @param x The input tensor.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static tanh<T extends Tensor>(x: T): T {
    return ENV.engine.executeKernel('Tanh', {inputs: {x}}, (dy: T, y: T) => {
      return {x: () => ops.scalar(1).sub(y.square()).mul(dy)};
    }) as T;
  }

  /**
   * Computes step of the input `Tensor` element-wise: `x > 0 ? 1 : alpha * x`
   *
   * @param x The input tensor.
   * @param alpha The gradient when input is negative.
   */
  @doc({heading: 'Operations', subheading: 'Basic math'})
  @operation
  static step<T extends Tensor>(x: T, alpha = 0.0): T {
    // TODO(manrajgrover): Return null for gradients when backprop supports it.
    return ENV.engine.executeKernel(
               'Step', {inputs: {x}, args: {alpha}}, (dy: T, y: T) => {
                 return {x: () => ops.zeros(y.shape)};
               }) as T;
  }
}

function preluDer(x: Tensor, alpha: Tensor): Tensor {
  return ENV.engine.executeKernel('PReLUDer', {inputs: {x, alpha}}) as Tensor;
}

function eluDer(x: Tensor): Tensor {
  return ENV.engine.executeKernel('EluDer', {inputs: {x}}) as Tensor;
}
