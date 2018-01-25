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
import * as broadcast_util from './broadcast_util';
import {operation} from './decorators';
import {NDArray, Scalar} from './ndarray';
import {DataType, Rank, RankMap} from './types';

export class Ops {
  /**
   * Adds two NDArrays element-wise, A + B. Supports broadcasting.
   * For a stricter version without broadcasting use addStrict().
   *
   * @param a The first `NDArray` to add.
   * @param b The second `NDArray` to add. Must have the same type as `a`.
   */
  @operation
  static add<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: NDArray<'float32'>, y: NDArray) => {
      const derA = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(a.shape);
      };
      const derB = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(b.shape);
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.executeKernel('Add', {inputs: {a, b}}, der) as T;
  }

  /**
   * Adds two NDArrays element-wise, A + B. Inputs must
   * be the same shape. For broadcasting support, use add() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */

  @operation
  static addStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
    return a.add(b);
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Supports broadcasting.
   * For a stricter version without broadcasting use subStrict().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static sub<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: NDArray<'float32'>, y: NDArray) => {
      const derA = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.reshape(a.shape);
      };
      const derB = () => {
        let res = dy;
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes);
        }
        return res.neg().reshape(b.shape);
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.executeKernel('Sub', {inputs: {a, b}}, der) as T;
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Inputs must
   * be the same shape. For broadcasting support, use sub() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  @operation
  static subStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
    return a.sub(b);
  }

  /**
   * Computes the power of one value to another. Supports broadcasting.
   * Given a tensor x and a tensor y, this operation computes x^y for
   * corresponding elements in x and y. For example:
   * x = tf.constant([[2, 2], [3, 3]])
   * y = tf.constant([[8, 16], [2, 3]])
   * pow(x, y)  # [[256, 65536], [9, 27]]
   *
   * @param base The base NDArray to pow element-wise.
   * @param exp The exponent NDArray to pow element-wise.
   */
  @operation
  static pow<D extends DataType, T extends NDArray<D>>(
      base: NDArray<D>, exp: NDArray<'int32'>): T {
    util.assert(
        exp.dtype === 'int32',
        'only supports int32 data type for the exponent parameter.');
    broadcast_util.assertAndGetBroadcastShape(base.shape, exp.shape);

    const gradient = (dy: NDArray<'float32'>, y: NDArray<D>) => {
      if (!util.arraysEqual(base.shape, exp.shape)) {
        throw new Error(
            `Gradient of pow not yet supported for broadcasted shapes.`);
      }
      const derBase = () => {
        const dx = exp.toFloat().mul(
            base.pow(exp.sub(Scalar.new(1, 'int32'))).toFloat());
        return dy.mul(dx);
      };
      const derExp = () => {
        throw new Error(
            `Backprop through exponent of math.pow not ` +
            `implemented yet.`);
      };
      return {base: derBase, exp: derExp};
    };

    return ENV.engine.executeKernel('Pow', {inputs: {base, exp}}, gradient) as
        T;
  }

  /**
   * Computes the power of one value to another. Inputs must
   * be the same shape. For broadcasting support, use pow() instead.
   *
   * @param base The base NDArray to pow element-wise.
   * @param exp The exponent NDArray to pow element-wise.
   */
  @operation
  static powStrict<D extends DataType, R extends Rank>(
      base: NDArray<D, R>, exp: NDArray<'int32'>): RankMap<D>[R] {
    util.assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
    return base.pow(exp);
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Supports broadcasting.
   * For a stricter version without broadcasting use mulStrict().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static mul<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

    const der = (dy: NDArray<'float32'>, y: NDArray) => {
      const derA = () => {
        const res = dy.mul(b.asType('float32'));
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(a.shape);
        }
        return res;
      };
      const derB = () => {
        const res = dy.mul(a.asType('float32'));
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(b.shape);
        }
        return res;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.executeKernel('Mul', {inputs: {a, b}}, der) as T;
  }

  /**
   * @deprecated Use mulStrict() instead.
   */
  @operation
  static elementWiseMul<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    return a.mulStrict(b);
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Inputs must
   * be the same shape. For broadcasting support, use mul().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static mulStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
    return a.mul(b) as RankMap<D>[R];
  }

  /**
   * Divides two NDArrays element-wise, A / B. Supports broadcasting.
   * For a stricter version without broadcasting use divStrict().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static div<D extends DataType, T extends NDArray<'float32'>>(
      a: NDArray<D>, b: NDArray<D>): T {
    const outShape =
        broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    const der = (dy: NDArray<'float32'>, y: NDArray) => {
      const derA = () => {
        const res = dy.div(b.asType('float32'));
        const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
        if (reduceAxes.length > 0) {
          return res.sum(reduceAxes).reshape(a.shape);
        }
        return res;
      };
      const derB = () => {
        let res = dy.mul(a.asType('float32'));
        const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
        if (reduceAxes.length > 0) {
          res = res.sum(reduceAxes).reshape(b.shape);
        }
        const tmp = b.square() as NDArray;
        return res.div(tmp.asType('float32')).neg() as NDArray<'float32'>;
      };
      return {a: derA, b: derB};
    };
    return ENV.engine.executeKernel('Div', {inputs: {a, b}}, der) as T;
  }

  /**
   * Divides two NDArrays element-wise, A / B. Inputs must
   * be the same shape. For broadcasting support, use div() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  @operation
  static divStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
    return a.div(b) as RankMap<D>[R];
  }

  /** @deprecated Use div() instead. */
  @operation
  static scalarDividedByArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarDividedByArray: first argument must be rank 0, but ` +
            `got NDArray of rank ${c.rank}.`);
    return c.div(a) as T;
  }

  /** @deprecated Use div(A, c) instead. */
  @operation
  static arrayDividedByScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: second argument must be rank 0, ` +
            `but got NDArray of rank ${c.rank}.`);
    return a.div(c) as T;
  }

  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * @param a The first ndarray.
   * @param b The second ndarray. Must have the same type as `a`.
   */
  @operation
  static minimum<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Minimum', {inputs: {a, b}}) as T;
  }

  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use minimum().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static minimumStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in minimumStrict: ');
    return a.minimum(b) as RankMap<D>[R];
  }

  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * @param a The first ndarray.
   * @param b The second ndarray. Must have the same type as `a`.
   */
  @operation
  static maximum<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('Maximum', {inputs: {a, b}}) as T;
  }

  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
   * be the same shape. For broadcasting support, use maximum().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  @operation
  static maximumStrict<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>): RankMap<D>[R] {
    util.assertShapesMatch(a.shape, b.shape, 'Error in minimumStrict: ');
    return a.maximum(b) as RankMap<D>[R];
  }
}
