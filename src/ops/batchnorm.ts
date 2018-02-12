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

import {operation} from './operation';
import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {Rank} from '../types';
import * as util from '../util';

export class Ops {
  /**
   * Batch normalization 2D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Tensor2D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Tensor1D of shape [depth].
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  @operation
  static batchNormalization2d(
      x: Tensor2D, mean: Tensor2D|Tensor1D, variance: Tensor2D|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor2D|Tensor1D,
      offset?: Tensor2D|Tensor1D): Tensor2D {
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
    }) as Tensor2D;
  }

  /**
   * Batch normalization 3D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Tensor3D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Tensor1D of shape [depth].
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  @operation
  static batchNormalization3d(
      x: Tensor3D, mean: Tensor3D|Tensor1D, variance: Tensor3D|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor3D|Tensor1D,
      offset?: Tensor3D|Tensor1D): Tensor3D {
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
    }) as Tensor3D;
  }

  /**
   * Batch normalization 4D. Mean, variance, scale, and offset can be of two
   * shapes: 1) The same shape as the input: an Tensor4D. 2) In the common
   * case, the depth dimension is the last dimension of x, so the values would
   * be an Tensor1D of shape [depth].
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  @operation
  static batchNormalization4d(
      x: Tensor4D, mean: Tensor4D|Tensor1D, variance: Tensor4D|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor4D|Tensor1D,
      offset?: Tensor4D|Tensor1D): Tensor4D {
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
    }) as Tensor4D;
  }

  @doc({heading: 'Operations', subheading: 'Normalization'})
  static batchNormalization<R extends Rank>(
      x: Tensor<R>, mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R> {
    if (x.rank === 0) {
      throw new Error(`Batchnorm for scalar is not supported`);
    } else if (x.rank === 1) {
      throw new Error(`Batchnorm for rank 1 is not yet implemented`);
    } else if (x.rank === 2) {
      return Ops.batchNormalization2d(
                 x as Tensor2D, mean as Tensor2D | Tensor1D,
                 variance as Tensor2D | Tensor1D, varianceEpsilon,
                 scale as Tensor2D | Tensor1D, offset as Tensor2D | Tensor1D) as
          Tensor<R>;
    } else if (x.rank === 3) {
      return Ops.batchNormalization3d(
                 x as Tensor3D, mean as Tensor3D | Tensor1D,
                 variance as Tensor3D | Tensor1D, varianceEpsilon,
                 scale as Tensor3D | Tensor1D, offset as Tensor3D | Tensor1D) as
          Tensor<R>;
    } else if (x.rank === 4) {
      return Ops.batchNormalization4d(
                 x as Tensor4D, mean as Tensor4D | Tensor1D,
                 variance as Tensor4D | Tensor1D, varianceEpsilon,
                 scale as Tensor4D | Tensor1D, offset as Tensor4D | Tensor1D) as
          Tensor<R>;
    } else {
      throw new Error(`Batchnorm for rank ${x.rank} is not yet implemented`);
    }
  }
}
