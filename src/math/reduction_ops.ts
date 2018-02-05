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
import {doc, operation} from './decorators';
import {Scalar, Tensor} from './tensor';

export class Ops {
  /**
   * Computes the log(sum(exp(elements across the reduction dimensions)).
   *
   * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param input The input Tensor.
   * @param axis Optional. The dimension(s) to reduce. If null (the default),
   *     reduces all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with length
   *     of 1. Defaults to false.
   */
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static logSumExp<T extends Tensor>(
      input: Tensor, axis: number|number[] = null, keepDims = false): T {
    const axes = axis_util.parseAxisParam(axis, input.shape);
    const xMax = input.max(axes, true /* keepDims */);
    const a = input.sub(xMax);
    const b = a.exp();
    const c = b.sum(axes);
    const d = c.log();
    const res = xMax.reshape(d.shape).add(d);

    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
      return res.reshape(newShape) as T;
    }
    return res as T;
  }

  /**
   * Computes the sum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If axes has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array to compute the sum over.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static sum<T extends Tensor>(
      x: Tensor, axis: number|number[] = null, keepDims = false): T {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    // Use a custom gradient to bypass 2 gradient backprops since sum is used
    // extremely often.
    return ENV.math.customGradient('sum', () => {
      const permutation = axis_util.getAxesPermutation(axes, x.rank);
      let reductionAxes = axes;
      let permutedX = x;
      if (permutation != null) {
        permutedX = x.transpose(permutation);
        reductionAxes =
            axis_util.getInnerMostAxes(reductionAxes.length, x.rank);
      }
      let value = ENV.engine.executeKernel(
          'Sum', {inputs: {x: permutedX}, args: {axes: reductionAxes}});
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
        value = value.reshape(newShape);
      }

      const gradients = (dy: Tensor) => {
        const expandedDyShape = x.shape.slice();
        axes.forEach(axis => {
          expandedDyShape[axis] = 1;
        });
        const expandedDy = dy.reshape(expandedDyShape);
        const derX = () => expandedDy.mul(Tensor.ones(x.shape, 'float32'));
        return {x: derX};
      };
      return {value, gradients};
    }, {x}) as T;
  }

  /**
   * Computes the mean of elements across dimensions of an array.
   *
   * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
   * true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static mean<T extends Tensor>(
      x: Tensor, axis: number|number[] = null, keepDims = false): T {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const shapes = axis_util.computeOutAndReduceShapes(x.shape, axes);
    const reduceShape = shapes[1];
    const reduceSize = util.sizeFromShape(reduceShape);
    // Use a custom gradient to bypass 2 gradient backprops since mean is used
    // extremely often.
    return ENV.math.customGradient('mean', () => {
      const reduceSizeScalar = Scalar.new(reduceSize);
      const res = x.div(reduceSizeScalar);
      const value = res.sum(axis, keepDims);

      const gradients = (dy: Tensor) => {
        const expandedDyShape = x.shape.slice();
        axes.forEach(axis => {
          expandedDyShape[axis] = 1;
        });
        const expandedDy = dy.reshape(expandedDyShape);
        const derX = () => expandedDy.mul(Tensor.ones(x.shape, 'float32'))
                               .div(reduceSizeScalar);
        return {x: derX};
      };
      return {value, gradients};
    }, {x}) as T;
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
   * @param x The input Tensor.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static min<T extends Tensor>(
      x: Tensor, axis: number|number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = x.transpose(permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }
    const res =
        ENV.engine.executeKernel('Min', {inputs: {x}, args: {axes}}) as Tensor;
    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
      return res.reshape(newShape) as T;
    }
    return res as T;
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
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static max<T extends Tensor>(
      x: Tensor, axis: number|number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = x.transpose(permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }
    const res =
        ENV.engine.executeKernel('Max', {inputs: {x}, args: {axes}}) as Tensor;
    if (keepDims) {
      const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
      return res.reshape(newShape) as T;
    }
    return res as T;
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
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static argMin<T extends Tensor>(x: Tensor, axis: number = null): T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = x.transpose(permutedAxes);
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
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static argMax<T extends Tensor>(x: Tensor, axis: number = null): T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getAxesPermutation(axes, x.rank);
    if (permutedAxes != null) {
      x = x.transpose(permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, x.rank);
    }

    return ENV.engine.executeKernel('ArgMax', {inputs: {x}, args: {axes}}) as T;
  }

  /**
   * Returns a 1 if the argMax of x1 and x2 are the same, otherwise 0.
   * @param x1 The first input Tensor.
   * @param x2 The second input Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Reduction'})
  @operation
  static argMaxEquals(x1: Tensor, x2: Tensor): Scalar {
    util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
    return x1.argMax().equal(x2.argMax());
  }

  /**
   * Calculates the mean and variance of `x`. The mean and variance are
   * calculated by aggregating the contents of `x` across `axes`. If `x` is
   * 1-D and `axes = [0]` this is just the mean and variance of a vector.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) along with to compute mean and
   *     variance. By default it reduces all dimensions.
   * @param keepDims If true, the moments have the same dimensionality as the
   *     input.
   * @return An object with two keys: `mean` and `variance`.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  @operation
  static moments(x: Tensor, axis: number|number[] = null, keepDims = false):
      {mean: Tensor, variance: Tensor} {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const mean = x.mean(axes, keepDims);
    let keepDimsShape = mean.shape;
    if (!keepDims) {
      keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
    }
    const devSquared = x.toFloat().sub(mean.reshape(keepDimsShape)).square();
    const variance = devSquared.mean(axes, keepDims);
    return {mean, variance};
  }
}
