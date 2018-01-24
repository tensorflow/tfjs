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

import {ENV} from '../environment';
import * as util from '../util';
import * as binary_ops from './binary_ops';
import {operation} from './decorators';
import {NDArray, Scalar} from './ndarray';
import {DataType, Rank} from './types';

export class Ops {
  /**
   * Computes -1 * A element-wise.
   * @param x The input array.
   */
  @operation
  static neg<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Neg', {inputs: {x}}) as T;
  }

  /**
   * Computes ceiling of input NDArray element-wise. y = ceil(x)
   * TODO(nsthorat): Make this return an int32 when we add rank as a
   * generic.
   * @param x The input NDArray.
   */
  @operation
  static ceil<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Ceil', {inputs: {x}}) as T;
  }

  /**
   * Computes floor of input NDArray element-wise. y = floor(x).
   *
   * @param x The input NDArray.
   */
  @operation
  static floor<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Floor', {inputs: {x}}) as T;
  }

  /**
   * Computes exponential of the input NDArray element-wise. y = e ^ x
   * @param x The input NDArray.
   */
  @operation
  static exp<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Exp', {inputs: {x}}) as T;
  }

  /**
   * Computes natural logarithm of the input NDArray element-wise. y = ln(x)
   * @param x The input NDArray.
   */
  @operation
  static log<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Log', {inputs: {x}}) as T;
  }

  /**
   * Computes square root of the input NDArray element-wise. y = sqrt(x)
   * @param x The input NDArray.
   */
  @operation
  static sqrt<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Sqrt', {inputs: {x}}) as T;
  }

  /**
   * Computes square of `x` element-wise.
   *
   * @param x The input array.
   */
  @operation
  static square<D extends DataType, R extends Rank, T extends NDArray<D, R>>(
      x: T): T {
    return ENV.engine.executeKernel(
               'Square', {inputs: {x}}, (dy: NDArray<'float32', R>, y: T) => {
                 return {
                   x: () => binary_ops.Ops.multiply(
                       dy,
                       binary_ops.Ops.multiply(
                           x.asType('float32'), Scalar.new(2)))
                 };
               }) as T;
  }

  /**
   * Computes absolute value element-wise.
   * @param x The input NDArray.
   */
  @operation
  static abs<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Abs', {inputs: {x}}) as T;
  }

  /**
   * Clips values element-wise.
   * @param x The input NDArray.
   * @param min Lower-bound of range to be clipped to.
   * @param max Upper-bound of range to be clipped to.
   */
  @operation
  static clip<T extends NDArray>(x: T, min: number, max: number): T {
    util.assert(
        (min <= max),
        `Error in clip: min (${min}) must be` +
            `less than or equal to max (${max}).`);
    return ENV.engine.executeKernel('Clip', {inputs: {x}, args: {min, max}}) as
        T;
  }

  /**
   * Computes rectified linear element-wise, max(x, 0).
   * @param x The input NDArray.
   */
  @operation
  static relu<D extends DataType, R extends Rank, T extends NDArray<D, R>>(
      x: T): T {
    return ENV.engine.executeKernel(
               'Relu', {inputs: {x}}, (dy: NDArray<'float32', R>, y: T) => {
                 return {
                   x: () => binary_ops.Ops.multiply(
                       dy, Ops.step(x).asType('float32'))
                 };
               }) as T;
  }

  /**
   * Computes exponential linear element-wise
   * @param x the input NDArray
   */
  @operation
  static elu<T extends NDArray>(x: T): T {
    const der = (dy: NDArray<'float32'>) => {
      return {
        x: () => binary_ops.Ops.multiply(dy, eluDer(x)),
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
   * @hidden
   */
  @operation
  static selu<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Selu', {inputs: {x}}) as T;
  }

  /**
   * Computes leaky rectified linear element-wise
   * @param x the input NDArray
   * @param alpha scaling factor for negative values, defaults to 0.2
   * @return {NDArray}
   */
  @operation
  static leakyRelu<T extends NDArray>(x: T, alpha = 0.2): T {
    return ENV.engine.executeKernel(
               'LeakyRelu', {inputs: {x}, args: {alpha}}) as T;
  }

  /**
   * Computes leaky rectified linear element-wise with parametric alphas
   * @param x the input NDArray
   * @param alpha scaling factor NDArray for negative values
   * @return {NDArray}
   */
  @operation
  static prelu<T extends NDArray>(x: T, alpha: T): T {
    const der = (dy: NDArray<'float32'>) => {
      return {
        x: () => binary_ops.Ops.multiply(dy, preluDer(x, alpha)),
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
   * Computes sigmoid element-wise, y = 1 / (1 + exp(-x)).
   * @param x The input NDArray.
   */
  @operation
  static sigmoid<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Sigmoid', {inputs: {x}}) as T;
  }

  /**
   * Computes sin of the input NDArray element-wise, y = sin(x).
   * @param x The input NDArray.
   */
  @operation
  static sin<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Sin', {inputs: {x}}) as T;
  }

  /**
   * Computes cos of the input NDArray element-wise, y = cos(x).
   * @param x The input NDArray.
   */
  @operation
  static cos<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Cos', {inputs: {x}}) as T;
  }

  /**
   * Computes tan of the input NDArray element-wise, y = tan(x).
   * @param x The input NDArray.
   */
  @operation
  static tan<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Tan', {inputs: {x}}) as T;
  }

  /**
   * Computes asin of the input NDArray element-wise, y = asin(x).
   * @param x The input NDArray.
   */
  @operation
  static asin<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Asin', {inputs: {x}}) as T;
  }

  /**
   * Computes acos of the input NDArray element-wise, y = acos(x).
   * @param x The input NDArray.
   */
  @operation
  static acos<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Acos', {inputs: {x}}) as T;
  }

  /**
   * Computes atan of the input NDArray element-wise, y = atan(x).
   * @param x The input NDArray.
   */
  @operation
  static atan<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Atan', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic sin of the input NDArray element-wise, y = sinh(x).
   * @param x The input NDArray.
   */
  @operation
  static sinh<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Sinh', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic cos of the input NDArray element-wise, y = cosh(x).
   * @param x The input NDArray.
   */
  @operation
  static cosh<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Cosh', {inputs: {x}}) as T;
  }

  /**
   * Computes hyperbolic tangent of the input NDArray element-wise.
   * @param x The input NDArray.
   */
  @operation
  static tanh<T extends NDArray>(x: T): T {
    return ENV.engine.executeKernel('Tanh', {inputs: {x}}) as T;
  }

  /**
   * Computes step of the input NDArray element-wise,
   * y=1 if x>0|alpha*x if x<=0.
   *
   * @param x The input NDArray.
   * @param alpha The gradient when input is negative.
   */
  @operation
  static step<T extends NDArray>(x: T, alpha = 0.0): T {
    return ENV.engine.executeKernel('Step', {inputs: {x}, args: {alpha}}) as T;
  }
}

function preluDer(x: NDArray, alpha: NDArray): NDArray<'float32'> {
  return ENV.engine.executeKernel('PReLUDer', {inputs: {x, alpha}}) as
      NDArray<'float32'>;
}

function eluDer(x: NDArray): NDArray<'float32'> {
  return ENV.engine.executeKernel('EluDer', {inputs: {x}}) as
      NDArray<'float32'>;
}
