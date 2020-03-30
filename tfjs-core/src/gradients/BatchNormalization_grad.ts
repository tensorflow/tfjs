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
import {BatchNormalization, BatchNormalizationAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {xAs4D} from '../ops/batchnorm_util';
import {getReductionAxes} from '../ops/broadcast_util';
import {scalar} from '../ops/tensor_ops';
import {tile} from '../ops/tile';
import {rsqrt} from '../ops/unary_ops';
import {Tensor, Tensor4D} from '../tensor';
import {Rank, ShapeMap} from '../types';

export const batchNormalizationGradConfig: GradConfig = {
  kernelName: BatchNormalization,
  inputsToSave: ['x', 'mean', 'variance', 'scale'],
  gradFunc: <R extends Rank>(
      dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const batchNormalizationAttrs: BatchNormalizationAttrs =
        attrs as {} as BatchNormalizationAttrs;
    const {varianceEpsilon} = batchNormalizationAttrs;
    const [x, mean, variance, scale] = saved;

    const x4D: Tensor4D = xAs4D(x);

    const scaleValue = scale == null ? scalar(1) : scale;
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
    const oneOverSqrtVariance = rsqrt(variance.add(scalar(varianceEpsilon)));
    const minusHalfRCube = oneOverSqrtVariance.mul(oneOverSqrtVariance)
                               .mul(oneOverSqrtVariance)
                               .mul(scalar(-0.5));

    const derX = () => {
      if (mean.rank === 1) {
        return dy
            .mul(tile(
                oneOverSqrtVariance.as4D(1, 1, 1, mean.shape[0]), tileShape))
            .mul(scaleValue)
            .reshape(x.shape);
      } else {
        return dy.mul(oneOverSqrtVariance).mul(scaleValue).reshape(x.shape);
      }
    };
    const derMean = () => {
      let meanDer = oneOverSqrtVariance.mul(scalar(-1)).mul(dyTimesScaleValue);
      if (mean.rank === 1) {
        meanDer = meanDer.sum(reductionAxes);
      }
      return meanDer.reshape(mean.shape as ShapeMap[R]);
    };
    const derVariance = () => {
      let varianceDer = minusHalfRCube.mul(xMinusMean).mul(dyTimesScaleValue);
      if (mean.rank === 1) {
        varianceDer = varianceDer.sum(reductionAxes);
      }
      return varianceDer.reshape(mean.shape as ShapeMap[R]);
    };
    const derScale = () => {
      const xMinusMean2TimesRsqrt = xMinusMean.mul(oneOverSqrtVariance);
      let scaleDer = dy.mul(xMinusMean2TimesRsqrt);
      if (mean.rank === 1) {
        scaleDer = scaleDer.sum(reductionAxes);
      }
      return scaleDer.reshape(mean.shape as ShapeMap[R]);
    };
    const derOffset = () => {
      let offsetDer = dy;
      if (mean.rank === 1) {
        offsetDer = offsetDer.sum(reductionAxes);
      }
      return offsetDer.reshape(mean.shape as ShapeMap[R]);
    };
    return {
      x: derX,
      mean: derMean,
      variance: derVariance,
      scale: derScale,
      offset: derOffset
    };
  }
};
