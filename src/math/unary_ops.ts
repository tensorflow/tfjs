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
import {operation} from './decorators';
import {NDArray, Scalar} from './ndarray';
import {DataType, Rank, RankMap} from './types';

export class Ops {
  /**
   * Computes -1 * A element-wise.
   * @param x The input array.
   */
  @operation
  static neg<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Neg', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes ceiling of input NDArray element-wise. y = ceil(x)
   * TODO(nsthorat): Make this return an int32 when we add rank as a
   * generic.
   * @param x The input NDArray.
   */
  @operation
  static ceil<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Ceil', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes floor of input NDArray element-wise. y = floor(x).
   *
   * @param x The input NDArray.
   */
  @operation
  static floor<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Floor', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes exponential of the input NDArray element-wise. y = e ^ x
   * @param x The input NDArray.
   */
  @operation
  static exp<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Exp', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes natural logarithm of the input NDArray element-wise. y = ln(x)
   * @param x The input NDArray.
   */
  @operation
  static log<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Log', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes square root of the input NDArray element-wise. y = sqrt(x)
   * @param x The input NDArray.
   */
  @operation
  static sqrt<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel(
               'Sqrt', {inputs: {x}},
               (dy: NDArray<'float32', R>, y: NDArray<D, R>) => {
                 return {
                   x: () =>
                       dy.div(x.asType('float32').sqrt().mul(Scalar.new(2)))
                 };
               }) as RankMap<D>[R];
  }

  /**
   * Computes square of `x` element-wise.
   *
   * @param x The input array.
   */
  @operation
  static square<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel(
               'Square', {inputs: {x}},
               (dy: NDArray<'float32', R>, y: RankMap<D>[R]) => {
                 return {
                   x: () => dy.mul(x.asType('float32').mul(Scalar.new(2)))
                 };
               }) as RankMap<D>[R];
  }

  /**
   * Computes absolute value element-wise.
   * @param x The input NDArray.
   */
  @operation
  static abs<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel(
               'Abs', {inputs: {x}},
               (dy: NDArray<'float32', R>, y: NDArray<D, R>) => {
                 return {x: () => dy.mul(x.toFloat().step(-1))};
               }) as RankMap<D>[R];
  }

  /**
   * Clips values element-wise.
   * @param x The input NDArray.
   * @param min Lower-bound of range to be clipped to.
   * @param max Upper-bound of range to be clipped to.
   */
  @operation
  static clip<D extends DataType, R extends Rank>(
      x: NDArray<D, R>, min: number, max: number): RankMap<D>[R] {
    util.assert(
        (min <= max),
        `Error in clip: min (${min}) must be` +
            `less than or equal to max (${max}).`);
    return ENV.engine.executeKernel('Clip', {inputs: {x}, args: {min, max}}) as
        RankMap<D>[R];
  }

  /**
   * Computes rectified linear element-wise, max(x, 0).
   * @param x The input NDArray.
   */
  @operation
  static relu<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel(
               'Relu', {inputs: {x}},
               (dy: NDArray<'float32', R>, y: RankMap<D>[R]) => {
                 const stepRes = x.step() as NDArray<'float32'>;
                 return {x: () => dy.mul(stepRes.asType('float32'))};
               }) as RankMap<D>[R];
  }

  /**
   * Computes exponential linear element-wise
   * @param x the input NDArray
   */
  @operation
  static elu<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    const der = (dy: NDArray<'float32'>) => {
      return {
        x: () => dy.mul(eluDer(x)),
        alpha: () => {
          throw new Error(
              'Derivative of prelu with respect to alpha is ' +
              'not implemented yet');
        }
      };
    };
    return ENV.engine.executeKernel('Elu', {inputs: {x}}, der) as RankMap<D>[R];
  }

  /**
   * Computes scaled exponential linear element-wise.
   * @hidden
   */
  @operation
  static selu<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Selu', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes leaky rectified linear element-wise
   * @param x the input NDArray
   * @param alpha scaling factor for negative values, defaults to 0.2
   * @return {NDArray}
   */
  @operation
  static leakyRelu<D extends DataType, R extends Rank>(
      x: NDArray<D, R>, alpha = 0.2): RankMap<D>[R] {
    return ENV.engine.executeKernel(
               'LeakyRelu', {inputs: {x}, args: {alpha}}) as RankMap<D>[R];
  }

  /**
   * Computes leaky rectified linear element-wise with parametric alphas
   * @param x the input NDArray
   * @param alpha scaling factor NDArray for negative values
   * @return {NDArray}
   */
  @operation
  static prelu<D extends DataType, R extends Rank>(
      x: NDArray<D, R>, alpha: NDArray<D, R>): RankMap<D>[R] {
    const der = (dy: NDArray<'float32'>) => {
      return {
        x: () => dy.mul(preluDer(x, alpha)),
        alpha: () => {
          throw new Error(
              'Derivative of prelu with respect to alpha is ' +
              'not implemented yet');
        }
      };
    };
    return ENV.engine.executeKernel('PReLU', {inputs: {x, alpha}}, der) as
        RankMap<D>[R];
  }

  /**
   * Computes sigmoid element-wise, y = 1 / (1 + exp(-x)).
   * @param x The input NDArray.
   */
  @operation
  static sigmoid<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Sigmoid', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes sin of the input NDArray element-wise, y = sin(x).
   * @param x The input NDArray.
   */
  @operation
  static sin<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Sin', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes cos of the input NDArray element-wise, y = cos(x).
   * @param x The input NDArray.
   */
  @operation
  static cos<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Cos', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes tan of the input NDArray element-wise, y = tan(x).
   * @param x The input NDArray.
   */
  @operation
  static tan<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Tan', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes asin of the input NDArray element-wise, y = asin(x).
   * @param x The input NDArray.
   */
  @operation
  static asin<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Asin', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes acos of the input NDArray element-wise, y = acos(x).
   * @param x The input NDArray.
   */
  @operation
  static acos<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Acos', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes atan of the input NDArray element-wise, y = atan(x).
   * @param x The input NDArray.
   */
  @operation
  static atan<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Atan', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes hyperbolic sin of the input NDArray element-wise, y = sinh(x).
   * @param x The input NDArray.
   */
  @operation
  static sinh<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Sinh', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes hyperbolic cos of the input NDArray element-wise, y = cosh(x).
   * @param x The input NDArray.
   */
  @operation
  static cosh<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Cosh', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes hyperbolic tangent of the input NDArray element-wise.
   * @param x The input NDArray.
   */
  @operation
  static tanh<D extends DataType, R extends Rank>(x: NDArray<D, R>):
      RankMap<D>[R] {
    return ENV.engine.executeKernel('Tanh', {inputs: {x}}) as RankMap<D>[R];
  }

  /**
   * Computes step of the input NDArray element-wise,
   * y=1 if x>0|alpha*x if x<=0.
   *
   * @param x The input NDArray.
   * @param alpha The gradient when input is negative.
   */
  @operation
  static step<D extends DataType, R extends Rank>(
      x: NDArray<D, R>, alpha = 0.0): RankMap<D>[R] {
    return ENV.engine.executeKernel('Step', {inputs: {x}, args: {alpha}}) as
        RankMap<D>[R];
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
