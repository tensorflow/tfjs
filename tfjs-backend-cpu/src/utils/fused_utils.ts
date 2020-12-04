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

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {elu} from '../kernels/Elu';
import {identity} from '../kernels/Identity';
import {leakyRelu} from '../kernels/LeakyRelu';
import {prelu} from '../kernels/Prelu';
import {relu} from '../kernels/Relu';
import {relu6} from '../kernels/Relu6';

export function applyActivation(
    backend: MathBackendCPU, x: TensorInfo, activation: backend_util.Activation,
    preluActivationWeights?: TensorInfo, leakyreluAlpha?: number): TensorInfo {
  if (activation === 'linear') {
    return identity({inputs: {x}, backend});
  } else if (activation === 'relu') {
    return relu({inputs: {x}, backend}) as TensorInfo;
  } else if (activation === 'elu') {
    return elu({inputs: {x}, backend}) as TensorInfo;
  } else if (activation === 'relu6') {
    return relu6({inputs: {x}, backend}) as TensorInfo;
  } else if (activation === 'prelu') {
    return prelu({inputs: {x, alpha: preluActivationWeights}, backend});
  } else if (activation === 'leakyrelu') {
    return leakyRelu({inputs: {x}, backend, attrs: {alpha: leakyreluAlpha}});
  }
  throw new Error(
      `Activation ${activation} has not been implemented for the CPU backend.`);
}
