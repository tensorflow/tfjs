/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {Tensor, Tensor3D, Tensor4D} from '../tensor';

import * as broadcast_util from './broadcast_util';
import {Conv2DInfo} from './conv_util';

export type Activation = 'linear'|'relu'|'prelu'|'elu'|'relu6';

export type FusedBatchMatMulConfig = {
  a: Tensor3D,
  b: Tensor3D,
  transposeA: boolean,
  transposeB: boolean,
  bias?: Tensor,
  activation?: Activation,
  preluActivationWeights?: Tensor
};

export type FusedConv2DConfig = {
  input: Tensor4D,
  filter: Tensor4D,
  convInfo: Conv2DInfo,
  bias?: Tensor,
  activation?: Activation,
  preluActivationWeights?: Tensor
};

// Returns gradient for fused activation.
export const getDyActivation =
    (dy: Tensor, y: Tensor, activation: Activation): Tensor => {
      if (activation == null || activation === 'linear') {
        return dy;
      }
      if (activation === 'relu') {
        return dy.mul(y.step());
      }
      throw new Error(
          `Gradient for activation ${activation} has not been ` +
          `implemented yet.`);
    };

// Returns gradient for fused bias.
export const getBiasGradient = (bias: Tensor, dyActivation: Tensor): Tensor => {
  let res = dyActivation;
  const reduceAxes =
      broadcast_util.getReductionAxes(bias.shape, dyActivation.shape);
  if (reduceAxes.length > 0) {
    res = res.sum(reduceAxes);
  }
  return res.reshape(bias.shape);
};

// Whether we should call non-fused ops instead.
export const shouldNotFuse =
    (gradientDepth: number, activation: Activation) => {
      const gradientMode = gradientDepth > 0;
      return gradientMode && activation !== 'linear' && activation !== 'relu';
    };
