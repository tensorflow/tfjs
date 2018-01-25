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
import {Array1D, Array2D, Array3D, Array4D, NDArray} from './ndarray';
import {DataType, Rank, RankMap} from './types';

export class Ops {
  /**
   * Batch normalization 2D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array2D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Array1D of shape [depth].
   * @param x The input NDArray.
   * @param mean A mean NDArray.
   * @param variance A variance NDArray.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale NDArray.
   * @param offset An offset NDArray.
   */
  @operation
  static batchNormalization2D<D extends DataType>(
      x: Array2D<D>, mean: Array2D<D>|Array1D<D>,
      variance: Array2D<D>|Array1D<D>, varianceEpsilon = .001,
      scale?: Array2D|Array1D, offset?: Array2D<D>|Array1D<D>): Array2D<D> {
    util.assert(
        x.rank === 2,
        `Error in batchNormalization3D: x must be rank 3 but got rank ` +
            `${x.rank}.`);
    util.assert(
        mean.rank === 2 || mean.rank === 1,
        `Error in batchNormalization2D: mean must be rank 2 or rank 1 but ` +
            `got rank ${mean.rank}.`);
    util.assert(
        variance.rank === 2 || variance.rank === 1,
        `Error in batchNormalization2D: variance must be rank 2 or rank 1 ` +
            `but got rank ${variance.rank}.`);
    if (scale != null) {
      util.assert(
          scale.rank === 2 || scale.rank === 1,
          `Error in batchNormalization2D: scale must be rank 2 or rank 1 ` +
              `but got rank ${scale.rank}.`);
    }
    if (offset != null) {
      util.assert(
          offset.rank === 2 || offset.rank === 1,
          `Error in batchNormalization2D: offset must be rank 2 or rank 1 ` +
              `but got rank ${offset.rank}.`);
    }

    return ENV.engine.executeKernel('BatchNorm2D', {
      inputs: {x, mean, variance, scale, offset},
      args: {varianceEpsilon}
    }) as Array2D<D>;
  }

  /**
   * Batch normalization 3D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array3D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Array1D of shape [depth].
   * @param x The input NDArray.
   * @param mean A mean NDArray.
   * @param variance A variance NDArray.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale NDArray.
   * @param offset An offset NDArray.
   */
  @operation
  static batchNormalization3D<D extends DataType>(
      x: Array3D<D>, mean: Array3D<D>|Array1D<D>,
      variance: Array3D<D>|Array1D<D>, varianceEpsilon = .001,
      scale?: Array3D<D>|Array1D<D>,
      offset?: Array3D<D>|Array1D<D>): Array3D<D> {
    util.assert(
        x.rank === 3,
        `Error in batchNormalization3D: x must be rank 3 but got rank ` +
            `${x.rank}.`);
    util.assert(
        mean.rank === 3 || mean.rank === 1,
        `Error in batchNormalization3D: mean must be rank 3 or rank 1 but ` +
            `got rank ${mean.rank}.`);
    util.assert(
        variance.rank === 3 || variance.rank === 1,
        `Error in batchNormalization3D: variance must be rank 3 or rank 1 ` +
            `but got rank ${variance.rank}.`);
    if (scale != null) {
      util.assert(
          scale.rank === 3 || scale.rank === 1,
          `Error in batchNormalization3D: scale must be rank 3 or rank 1 ` +
              `but got rank ${scale.rank}.`);
    }
    if (offset != null) {
      util.assert(
          offset.rank === 3 || offset.rank === 1,
          `Error in batchNormalization3D: offset must be rank 3 or rank 1 ` +
              `but got rank ${offset.rank}.`);
    }

    return ENV.engine.executeKernel('BatchNorm3D', {
      inputs: {x, mean, variance, scale, offset},
      args: {varianceEpsilon}
    }) as Array3D<D>;
  }

  /**
   * Batch normalization 4D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Array4D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Array1D of shape [depth].
   * @param x The input NDArray.
   * @param mean A mean NDArray.
   * @param variance A variance NDArray.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale NDArray.
   * @param offset An offset NDArray.
   */
  @operation
  static batchNormalization4D<D extends DataType>(
      x: Array4D<D>, mean: Array4D<D>|Array1D<D>,
      variance: Array4D<D>|Array1D<D>, varianceEpsilon = .001,
      scale?: Array4D<D>|Array1D<D>,
      offset?: Array4D<D>|Array1D<D>): Array4D<D> {
    util.assert(
        x.rank === 4,
        `Error in batchNormalization4D: x must be rank 4 but got rank ` +
            `${x.rank}.`);
    util.assert(
        mean.rank === 4 || mean.rank === 1,
        `Error in batchNormalization4D: mean must be rank 4 or rank 1 but ` +
            `got rank ${mean.rank}.`);
    util.assert(
        variance.rank === 4 || variance.rank === 1,
        `Error in batchNormalization4D: variance must be rank 4 or rank 1 ` +
            `but got rank ${variance.rank}.`);
    if (scale != null) {
      util.assert(
          scale.rank === 4 || scale.rank === 1,
          `Error in batchNormalization4D: scale must be rank 4 or rank 1 ` +
              `but got rank ${scale.rank}.`);
    }
    if (offset != null) {
      util.assert(
          offset.rank === 4 || offset.rank === 1,
          `Error in batchNormalization4D: offset must be rank 4 or rank 1 ` +
              `but got rank ${offset.rank}.`);
    }

    return ENV.engine.executeKernel('BatchNorm4D', {
      inputs: {x, mean, variance, scale, offset},
      args: {varianceEpsilon}
    }) as Array4D<D>;
  }

  static batchNormalization<D extends DataType, R extends Rank>(
      x: NDArray<D, R>, mean: RankMap<D>[R]|Array1D,
      variance: RankMap<D>[R]|Array1D, varianceEpsilon = .001,
      scale?: RankMap<D>[R]|Array1D,
      offset?: RankMap<D>[R]|Array1D): RankMap<D>[R] {
    if (x.rank === 0) {
      throw new Error(`Batchnorm for scalar is not supported`);
    } else if (x.rank === 1) {
      throw new Error(`Batchnorm for rank 1 is not yet implemented`);
    } else if (x.rank === 2) {
      return Ops.batchNormalization2D(
          x as Array2D<D>, mean as Array2D<D>| Array1D<D>,
          variance as Array2D<D>| Array1D<D>, varianceEpsilon,
          scale as Array2D<D>| Array1D<D>, offset as Array2D<D>| Array1D<D>);
    } else if (x.rank === 3) {
      return Ops.batchNormalization3D(
          x as Array3D<D>, mean as Array3D<D>| Array1D<D>,
          variance as Array3D<D>| Array1D<D>, varianceEpsilon,
          scale as Array3D<D>| Array1D<D>, offset as Array3D<D>| Array1D<D>);
    } else if (x.rank === 4) {
      return Ops.batchNormalization4D(
          x as Array4D<D>, mean as Array4D<D>| Array1D<D>,
          variance as Array4D<D>| Array1D<D>, varianceEpsilon,
          scale as Array4D<D>| Array1D<D>, offset as Array4D<D>| Array1D<D>);
    } else {
      throw new Error(`Batchnorm for rank ${x.rank} is not yet implemented`);
    }
  }
}
