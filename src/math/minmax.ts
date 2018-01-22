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
import * as axis_util from './axis_util';
import * as broadcast_util from './broadcast_util';
import * as compare from './compare';
import {operation} from './decorators';
import {DataType, NDArray, Scalar} from './ndarray';
import * as transpose from './transpose';

export class Ops {
  /**
   * Computes the minimum value from the input.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input NDArray.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  @operation
  static min<D extends DataType, T extends NDArray<D>>(
      x: NDArray<D>, axis: number|number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = transpose.Ops.transpose(x, permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }
    const res = ENV.engine.executeKernel('Min', {inputs: {x}, args: {axes}}) as
        NDArray<D>;
    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
      return res.reshape(newShape) as T;
    }
    return res as T;
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
   * Computes the maximum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  @operation
  static max<D extends DataType, T extends NDArray<D>>(
      x: NDArray<D>, axis: number|number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = transpose.Ops.transpose(x, permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }
    const res = ENV.engine.executeKernel('Max', {inputs: {x}, args: {axes}}) as
        NDArray<D>;
    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
      return res.reshape(newShape) as T;
    }
    return res as T;
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
   * Returns the indices of the minimum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   * across all axes and returns the flat index.
   *
   */
  @operation
  static argMin<T extends NDArray<'int32'>>(x: NDArray, axis: number = null):
      T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = transpose.Ops.transpose(x, permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }
    return ENV.engine.executeKernel('ArgMin', {inputs: {x}, args: {axes}}) as T;
  }

  /**
   * Returns the indices of the maximum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   *     across all axes and returns the flat index
   */
  @operation
  static argMax<T extends NDArray<'int32'>>(x: NDArray, axis: number = null):
      T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = transpose.Ops.transpose(x, permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }

    return ENV.engine.executeKernel('ArgMax', {inputs: {x}, args: {axes}}) as T;
  }

  /**
   * Returns a 1 if the argMax of x1 and x2 are the same, otherwise 0.
   * @param x1 The first input NDArray.
   * @param x2 The second input NDArray.
   */
  @operation
  static argMaxEquals(x1: NDArray, x2: NDArray): Scalar<'bool'> {
    util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
    return compare.Ops.equal(Ops.argMax(x1), Ops.argMax(x2));
  }
}
