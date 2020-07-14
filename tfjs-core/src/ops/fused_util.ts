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

import {Tensor} from '../tensor';

import * as broadcast_util from './broadcast_util';
import {elu} from './elu';
import {Activation} from './fused_types';
import {prelu} from './prelu';
import {relu} from './relu';
import {relu6} from './relu6';

// Returns gradient for fused activation.
export function getFusedDyActivation(
    dy: Tensor, y: Tensor, activation: Activation): Tensor {
  if (activation == null || activation === 'linear') {
    return dy;
  }
  if (activation === 'relu') {
    return dy.mul(y.step());
  }
  throw new Error(
      `Gradient for activation ${activation} has not been ` +
      `implemented yet.`);
}

// Returns gradient for fused bias.
export function getFusedBiasGradient(
    bias: Tensor, dyActivation: Tensor): Tensor {
  let res = dyActivation;
  const reduceAxes =
      broadcast_util.getReductionAxes(bias.shape, dyActivation.shape);
  if (reduceAxes.length > 0) {
    res = res.sum(reduceAxes);
  }
  return res.reshape(bias.shape);
}

export function applyActivation(
    x: Tensor, activation: Activation,
    preluActivationWeights?: Tensor): Tensor {
  if (activation === 'linear') {
    return x;
  } else if (activation === 'relu') {
    return relu(x);
  } else if (activation === 'elu') {
    return elu(x);
  } else if (activation === 'relu6') {
    return relu6(x);
  } else if (activation === 'prelu') {
    return prelu(x, preluActivationWeights);
  }
  throw new Error(`Unknown fused activation ${activation}.`);
}
