/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {FusedBatchNorm, FusedBatchNormAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import {add} from '../ops/add';
import {getReductionAxes} from '../ops/broadcast_util';
import {mul} from '../ops/mul';
import {reshape} from '../ops/reshape';
import {rsqrt} from '../ops/rsqrt';
import {scalar} from '../ops/scalar';
import {sub} from '../ops/sub';
import {sum} from '../ops/sum';
import {tile} from '../ops/tile';
import {Tensor} from '../tensor';
import {Rank, ShapeMap} from '../types';

export const fusedBatchNormGradConfig: GradConfig = {
  kernelName: FusedBatchNorm,
  inputsToSave: ['x', 'mean', 'variance', 'scale'],
  gradFunc: <R extends Rank>(
      dy: Tensor, saved: Tensor[], attrs: NamedAttrMap) => {
    const {varianceEpsilon} = attrs as {} as FusedBatchNormAttrs;
    const [x, mean, variance, scale] = saved;

    const scaleValue = scale == null ? scalar(1) : scale;
    const reductionAxes = getReductionAxes(mean.shape, x.shape);
    const tileShape: number[] = [];
    if (mean.rank === 1) {
      for (let i = 0; i < x.shape.length - 1; ++i) {
        tileShape.push(x.shape[i]);
      }
      tileShape.push(1);
    }

    const xMinusMean = sub(x, mean);
    const dyTimesScaleValue = mul(dy, scaleValue);
    const oneOverSqrtVariance = rsqrt(add(variance, scalar(varianceEpsilon)));
    const minusHalfRCube = mul(
        mul(mul(oneOverSqrtVariance, oneOverSqrtVariance), oneOverSqrtVariance),
        scalar(-0.5));

    const derX = () => {
      if (mean.rank === 1) {
        return reshape(
            mul(mul(dy,
                    tile(
                        reshape(oneOverSqrtVariance, [1, 1, 1, mean.shape[0]]),
                        tileShape)),
                scaleValue),
            x.shape);
      } else {
        return reshape(mul(mul(dy, oneOverSqrtVariance), scaleValue), x.shape);
      }
    };
    const derMean = () => {
      let meanDer =
          mul(mul(oneOverSqrtVariance, scalar(-1)), dyTimesScaleValue);
      if (mean.rank === 1) {
        meanDer = sum(meanDer, reductionAxes);
      }
      return reshape(meanDer, mean.shape as ShapeMap[R]);
    };
    const derVariance = () => {
      let varianceDer = mul(mul(minusHalfRCube, xMinusMean), dyTimesScaleValue);

      if (mean.rank === 1) {
        varianceDer = sum(varianceDer, reductionAxes);
      }
      return reshape(varianceDer, mean.shape as ShapeMap[R]);
    };
    const derScale = () => {
      const xMinusMean2TimesRsqrt = mul(xMinusMean, oneOverSqrtVariance);

      let scaleDer = mul(dy, xMinusMean2TimesRsqrt);
      if (mean.rank === 1) {
        scaleDer = sum(scaleDer, reductionAxes);
      }
      return reshape(scaleDer, mean.shape as ShapeMap[R]);
    };
    const derOffset = () => {
      let offsetDer = dy;
      if (mean.rank === 1) {
        offsetDer = sum(offsetDer, reductionAxes);
      }
      return reshape(offsetDer, mean.shape as ShapeMap[R]);
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
