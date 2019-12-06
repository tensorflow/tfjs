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

import {ENGINE} from '../engine';
import {deprecationWarn} from '../globals';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';
import * as util from '../util';

import {tile} from './array_ops';
import {getReductionAxes} from './broadcast_util';
import {op} from './operation';
import {scalar} from './tensor_ops';
import {rsqrt} from './unary_ops';

/**
 * Batch normalization, strictly for 2D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm2d_(
    x: Tensor2D|TensorLike, mean: Tensor2D|Tensor1D|TensorLike,
    variance: Tensor2D|Tensor1D|TensorLike,
    offset?: Tensor2D|Tensor1D|TensorLike, scale?: Tensor2D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor2D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor2D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor2D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 2,
      () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 2 || $mean.rank === 1,
      () => `Error in batchNorm2D: mean must be rank 2 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 2 || $variance.rank === 1,
      () => `Error in batchNorm2D: variance must be rank 2 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 2 || $scale.rank === 1,
        () => `Error in batchNorm2D: scale must be rank 2 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 2 || $offset.rank === 1,
        () => `Error in batchNorm2D: offset must be rank 2 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }

  return batchNorm_($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

/**
 * Batch normalization, strictly for 3D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm3d_(
    x: Tensor3D|TensorLike, mean: Tensor3D|Tensor1D|TensorLike,
    variance: Tensor3D|Tensor1D|TensorLike,
    offset?: Tensor3D|Tensor1D|TensorLike, scale?: Tensor3D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor3D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor3D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor3D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 3,
      () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 3 || $mean.rank === 1,
      () => `Error in batchNorm3D: mean must be rank 3 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 3 || $variance.rank === 1,
      () => `Error in batchNorm3D: variance must be rank 3 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 3 || $scale.rank === 1,
        () => `Error in batchNorm3D: scale must be rank 3 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 3 || $offset.rank === 1,
        () => `Error in batchNorm3D: offset must be rank 3 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }

  return batchNorm_($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

/**
 * Batch normalization, strictly for 4D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm4d_(
    x: Tensor4D|TensorLike, mean: Tensor4D|Tensor1D|TensorLike,
    variance: Tensor4D|Tensor1D|TensorLike,
    offset?: Tensor4D|Tensor1D|TensorLike, scale?: Tensor4D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor4D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor4D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor4D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 4,
      () => `Error in batchNorm4D: x must be rank 4 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 4 || $mean.rank === 1,
      () => `Error in batchNorm4D: mean must be rank 4 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 4 || $variance.rank === 1,
      () => `Error in batchNorm4D: variance must be rank 4 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 4 || $scale.rank === 1,
        () => `Error in batchNorm4D: scale must be rank 4 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 4 || $offset.rank === 1,
        () => `Error in batchNorm4D: offset must be rank 4 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }
  return batchNorm_($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

/**
 * @deprecated Please use `tf.batchNorm` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization_<R extends Rank>(
    x: Tensor<R>|TensorLike, mean: Tensor<R>|Tensor1D|TensorLike,
    variance: Tensor<R>|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor<R>|Tensor1D|TensorLike,
    offset?: Tensor<R>|Tensor1D|TensorLike): Tensor<R> {
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
    x: Tensor<R>|TensorLike, mean: Tensor<R>|Tensor1D|TensorLike,
    variance: Tensor<R>|Tensor1D|TensorLike,
    offset?: Tensor<R>|Tensor1D|TensorLike,
    scale?: Tensor<R>|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor<R> {
  if (varianceEpsilon == null) {
    varianceEpsilon = 0.001;
  }
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor<R>|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor<R>|Tensor1D;
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
      Tensor<R>, Tensor<R>| Tensor1D, Tensor<R>| Tensor1D, Tensor<R>| Tensor1D
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

/**
 * @deprecated Please use `tf.batchNorm2d` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization2d_(
    x: Tensor2D|TensorLike, mean: Tensor2D|Tensor1D|TensorLike,
    variance: Tensor2D|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor2D|Tensor1D|TensorLike,
    offset?: Tensor2D|Tensor1D|TensorLike): Tensor2D {
  warnDeprecation();
  return batchNorm2d_(x, mean, variance, offset, scale, varianceEpsilon);
}

/**
 * @deprecated Please use `tf.batchNorm3d` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization3d_(
    x: Tensor3D|TensorLike, mean: Tensor3D|Tensor1D|TensorLike,
    variance: Tensor3D|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor3D|Tensor1D|TensorLike,
    offset?: Tensor3D|Tensor1D|TensorLike): Tensor3D {
  warnDeprecation();
  return batchNorm3d_(x, mean, variance, offset, scale, varianceEpsilon);
}

/**
 * @deprecated Please use `tf.batchNorm4d` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization4d_(
    x: Tensor4D|TensorLike, mean: Tensor4D|Tensor1D|TensorLike,
    variance: Tensor4D|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor4D|Tensor1D|TensorLike,
    offset?: Tensor4D|Tensor1D|TensorLike): Tensor4D {
  warnDeprecation();
  return batchNorm4d_(x, mean, variance, offset, scale, varianceEpsilon);
}

function warnDeprecation() {
  deprecationWarn(
      'tf.batchNormalization() is going away. ' +
      'Use tf.batchNorm() instead, and note the positional argument change ' +
      'of scale, offset, and varianceEpsilon');
}

export const batchNormalization2d = op({batchNormalization2d_});
export const batchNormalization3d = op({batchNormalization3d_});
export const batchNormalization4d = op({batchNormalization4d_});
export const batchNormalization = op({batchNormalization_});

export const batchNorm = op({batchNorm_});
export const batchNorm2d = op({batchNorm2d_});
export const batchNorm3d = op({batchNorm3d_});
export const batchNorm4d = op({batchNorm4d_});
