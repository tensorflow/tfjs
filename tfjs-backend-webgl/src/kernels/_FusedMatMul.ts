/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {batchMatMulImpl} from './BatchMatMul_impl';

export function _fusedMatMul(args: {
  inputs: _FusedMatMulInputs,
  attrs: _FusedMatMulAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {a, b, bias, preluActivationWeights} = inputs;
  const {transposeA, transposeB, activation, leakyreluAlpha} = attrs;

  return batchMatMulImpl({
    a,
    b,
    transposeA,
    transposeB,
    backend,
    bias,
    preluActivationWeights,
    leakyreluAlpha,
    activation
  });
}

export const _fusedMatMulConfig: KernelConfig = {
  kernelName: _FusedMatMul,
  backendName: 'webgl',
  kernelFunc: _fusedMatMul as {} as KernelFunc,
};
