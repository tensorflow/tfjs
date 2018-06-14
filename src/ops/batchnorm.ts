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
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {Rank} from '../types';
import * as util from '../util';

import {ArrayOps} from './array_ops';
import {getReductionAxes} from './broadcast_util';
import {operation} from './operation';
import {rsqrt} from './ops';

export class BatchNormOps {
  /**
   * Batch normalization, strictly for 2D. For the more relaxed version, see
   * `batchNormalization`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
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

    return BatchNormOps.batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset);
  }

  /**
   * Batch normalization, strictly for 3D. For the more relaxed version, see
   * `batchNormalization`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
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

    return BatchNormOps.batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset);
  }

  /**
   * Batch normalization, strictly for 4D. For the more relaxed version, see
   * `batchNormalization`.
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
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
    return BatchNormOps.batchNormalization(
        x, mean, variance, varianceEpsilon, scale, offset);
  }

  /**
   * Batch normalization.
   *
   * As described in
   * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
   *
   * Mean, variance, scale, and offset can be of two
   * shapes:
   *   - The same shape as the input.
   *   - In the common case, the depth dimension is the last dimension of x, so
   *     the values would be an `Tensor1D` of shape [depth].
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that parameters passed are of given rank
   *   - `tf.batchNormalization2d`
   *   - `tf.batchNormalization3d`
   *   - `tf.batchNormalization4d`
   *
   * @param x The input Tensor.
   * @param mean A mean Tensor.
   * @param variance A variance Tensor.
   * @param varianceEpsilon A small float number to avoid dividing by 0.
   * @param scale A scale Tensor.
   * @param offset An offset Tensor.
   */
  @doc({heading: 'Operations', subheading: 'Normalization'})
  static batchNormalization<R extends Rank>(
      x: Tensor<R>, mean: Tensor<R>|Tensor1D, variance: Tensor<R>|Tensor1D,
      varianceEpsilon = .001, scale?: Tensor<R>|Tensor1D,
      offset?: Tensor<R>|Tensor1D): Tensor<R> {
    util.assertArgumentsAreTensors({x, mean, variance}, 'batchNormalization');
    if (scale != null) {
      util.assertArgumentsAreTensors({scale}, 'batchNormalization');
    }
    if (offset != null) {
      util.assertArgumentsAreTensors({offset}, 'batchNormalization');
    }

    util.assert(
        mean.rank === variance.rank,
        'Batch normalization gradient requires mean and variance to have ' +
            'equal ranks.');
    util.assert(
        offset == null || mean.rank === offset.rank,
        'Batch normalization gradient requires mean and offset to have ' +
            'equal ranks.');
    util.assert(
        scale == null || mean.rank === scale.rank,
        'Batch normalization gradient requires mean and scale to have ' +
            'equal ranks.');

    let x4D: Tensor4D;
    if (x.rank === 0 || x.rank === 1) {
      x4D = x.as4D(1, 1, 1, x.size);
    } else if (x.rank === 2) {
      x4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
    } else if (x.rank === 3) {
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]) as Tensor4D;
    } else {
      x4D = x as Tensor4D;
    }

    const der = (dy: Tensor) => {
      const scaleValue = scale == null ? ArrayOps.scalar(1) : scale;
      const reductionAxes = getReductionAxes(mean.shape, x4D.shape);
      const tileShape: number[] = [];
      if (mean.rank === 1) {
        for (let i = 0; i < x4D.shape.length - 1; ++i) {
          tileShape.push(x4D.shape[i]);
        }
        tileShape.push(1);
      }

      const xMinusMean = x.sub(mean);
      const dyTimesScaleValue = dy.mul(scaleValue);
      const oneOverSqrtVariance =
          rsqrt(variance.add(ArrayOps.scalar(varianceEpsilon)));
      const minusHalfRCube = oneOverSqrtVariance.mul(oneOverSqrtVariance)
                                 .mul(oneOverSqrtVariance)
                                 .mul(ArrayOps.scalar(-0.5));
      const derX = () => {
        if (mean.rank === 1) {
          return dy
              .mul(ArrayOps.tile(
                  oneOverSqrtVariance.as4D(1, 1, 1, mean.shape[0]), tileShape))
              .mul(scaleValue)
              .reshape(x.shape);
        } else {
          return dy.mul(oneOverSqrtVariance).mul(scaleValue).reshape(x.shape);
        }
      };
      const derMean = () => {
        let meanDer =
            oneOverSqrtVariance.mul(ArrayOps.scalar(-1)).mul(dyTimesScaleValue);
        if (mean.rank === 1) {
          meanDer = meanDer.sum(reductionAxes);
        }
        return meanDer.reshape(mean.shape);
      };
      const derVariance = () => {
        let varianceDer = minusHalfRCube.mul(xMinusMean).mul(dyTimesScaleValue);
        if (mean.rank === 1) {
          varianceDer = varianceDer.sum(reductionAxes);
        }
        return varianceDer.reshape(mean.shape);
      };
      const derScale = () => {
        const xMinusMean2TimesRsqrt = xMinusMean.mul(oneOverSqrtVariance);
        let scaleDer = dy.mul(xMinusMean2TimesRsqrt);
        if (mean.rank === 1) {
          scaleDer = scaleDer.sum(reductionAxes);
        }
        return scaleDer.reshape(mean.shape);
      };
      const derOffset = () => {
        let offsetDer = dy;
        if (mean.rank === 1) {
          offsetDer = offsetDer.sum(reductionAxes);
        }
        return offsetDer.reshape(mean.shape);
      };
      return {
        x: derX,
        mean: derMean,
        variance: derVariance,
        scale: derScale,
        offset: derOffset
      };
    };

    const res = ENV.engine.runKernel(
        backend => backend.batchNormalization(
            x4D, batchnormReshape4D(mean), batchnormReshape4D(variance),
            varianceEpsilon, batchnormReshape4D(scale),
            batchnormReshape4D(offset)),
        {x, mean, variance, scale, offset}, der);
    return res.reshape(x.shape);
  }
}

function batchnormReshape4D(x: Tensor): Tensor4D|Tensor1D {
  if (x == null) {
    return null;
  }
  if (x.rank === 0) {
    return x.as1D();
  } else if (x.rank === 1) {
    return x as Tensor1D;
  } else if (x.rank === 2) {
    return x.as4D(1, 1, x.shape[0], x.shape[1]);
  } else if (x.rank === 3) {
    return x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
  }
  return x as Tensor4D;
}
