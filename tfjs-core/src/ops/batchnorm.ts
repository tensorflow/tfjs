/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {Tensor, Tensor1D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';
import * as util from '../util';

import {getReductionAxes} from './broadcast_util';
import {op} from './operation';
import {scalar} from './tensor_ops';
import {tile} from './tile';
import {rsqrt} from './unary_ops';
import {warnDeprecation} from './batchnorm_util';

/**
 * @deprecated Please use `tf.batchNorm` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization_<R extends Rank>(
  x: Tensor<R> | TensorLike, mean: Tensor<R> | Tensor1D | TensorLike,
  variance: Tensor<R> | Tensor1D | TensorLike, varianceEpsilon = .001,
  scale?: Tensor<R> | Tensor1D | TensorLike,
  offset?: Tensor<R> | Tensor1D | TensorLike): Tensor<R> {
  warnDeprecation();
  return batchNorm_(x, mean, variance, offset, scale, varianceEpsilon);
}

/**
 * Batch normalization.
 *
 * As described in
 * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
 *
 * Mean, variance, scale, and offset can be of two shapes:
 *   - The same shape as the input.
 *   - In the common case, the depth dimension is the last dimension of x, so
 *     the values would be an `tf.Tensor1D` of shape [depth].
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that parameters passed are of given rank
 *   - `tf.batchNorm2d`
 *   - `tf.batchNorm3d`
 *   - `tf.batchNorm4d`
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
/** @doc {heading: 'Operations', subheading: 'Normalization'} */
function batchNorm_<R extends Rank>(
  x: Tensor<R> | TensorLike, mean: Tensor<R> | Tensor1D | TensorLike,
  variance: Tensor<R> | Tensor1D | TensorLike,
  offset?: Tensor<R> | Tensor1D | TensorLike,
  scale?: Tensor<R> | Tensor1D | TensorLike,
  varianceEpsilon?: number): Tensor<R> {
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor<R> | Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor<R> | Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }

  util.assert(
    $mean.rank === $variance.rank,
    () => 'Batch normalization gradient requires mean and variance to have ' +
      'equal ranks.');
  util.assert(
    $offset == null || $mean.rank === $offset.rank,
    () => 'Batch normalization gradient requires mean and offset to have ' +
      'equal ranks.');
  util.assert(
    $scale == null || $mean.rank === $scale.rank,
    () => 'Batch normalization gradient requires mean and scale to have ' +
      'equal ranks.');

  let x4D: Tensor4D;
  if ($x.rank === 0 || $x.rank === 1) {
    x4D = $x.as4D(1, 1, 1, $x.size);
  } else if ($x.rank === 2) {
    x4D = $x.as4D(1, 1, $x.shape[0], $x.shape[1]);
  } else if ($x.rank === 3) {
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  } else {
    x4D = $x as Tensor4D;
  }

  const der = (dy: Tensor, saved: Tensor[]) => {
    type Saved = [
      Tensor<R>, Tensor<R> | Tensor1D, Tensor<R> | Tensor1D, Tensor<R> | Tensor1D
    ];
    const [$x, $mean, $variance, $scale] = saved as Saved;
    const scaleValue = $scale == null ? scalar(1) : $scale;
    const reductionAxes = getReductionAxes($mean.shape, x4D.shape);
    const tileShape: number[] = [];
    if ($mean.rank === 1) {
      for (let i = 0; i < x4D.shape.length - 1; ++i) {
        tileShape.push(x4D.shape[i]);
      }
      tileShape.push(1);
    }

    const xMinusMean = $x.sub($mean);
    const dyTimesScaleValue = dy.mul(scaleValue);
    const oneOverSqrtVariance = rsqrt($variance.add(scalar(varianceEpsilon)));
    const minusHalfRCube = oneOverSqrtVariance.mul(oneOverSqrtVariance)
      .mul(oneOverSqrtVariance)
      .mul(scalar(-0.5));

    const derX = () => {
      if ($mean.rank === 1) {
        return dy
          .mul(tile(
            oneOverSqrtVariance.as4D(1, 1, 1, $mean.shape[0]), tileShape))
          .mul(scaleValue)
          .reshape($x.shape);
      } else {
        return dy.mul(oneOverSqrtVariance).mul(scaleValue).reshape($x.shape);
      }
    };
    const derMean = () => {
      let meanDer = oneOverSqrtVariance.mul(scalar(-1)).mul(dyTimesScaleValue);
      if ($mean.rank === 1) {
        meanDer = meanDer.sum(reductionAxes);
      }
      return meanDer.reshape($mean.shape as ShapeMap[R]);
    };
    const derVariance = () => {
      let varianceDer = minusHalfRCube.mul(xMinusMean).mul(dyTimesScaleValue);
      if ($mean.rank === 1) {
        varianceDer = varianceDer.sum(reductionAxes);
      }
      return varianceDer.reshape($mean.shape as ShapeMap[R]);
    };
    const derScale = () => {
      const xMinusMean2TimesRsqrt = xMinusMean.mul(oneOverSqrtVariance);
      let scaleDer = dy.mul(xMinusMean2TimesRsqrt);
      if ($mean.rank === 1) {
        scaleDer = scaleDer.sum(reductionAxes);
      }
      return scaleDer.reshape($mean.shape as ShapeMap[R]);
    };
    const derOffset = () => {
      let offsetDer = dy;
      if ($mean.rank === 1) {
        offsetDer = offsetDer.sum(reductionAxes);
      }
      return offsetDer.reshape($mean.shape as ShapeMap[R]);
    };
    return {
      x: derX,
      mean: derMean,
      variance: derVariance,
      scale: derScale,
      offset: derOffset
    };
  };

  const inputsToSave = [$x, $mean, $variance, $scale];

  const res = ENGINE.runKernelFunc(
    (backend, save) => {
      const res = backend.batchNormalization(
        x4D, batchnormReshape4D($mean), batchnormReshape4D($variance),
        varianceEpsilon, batchnormReshape4D($scale),
        batchnormReshape4D($offset));
      save([$x, $mean, $variance, $scale]);
      return res;
    },
    {x: $x, mean: $mean, variance: $variance, scale: $scale, offset: $offset},
    der, 'BatchNormalization', {varianceEpsilon}, inputsToSave);
  return res.reshape($x.shape);
}

function batchnormReshape4D(x: Tensor): Tensor4D | Tensor1D {
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

export const batchNormalization = op({batchNormalization_});
export const batchNorm = op({batchNorm_});
