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

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {applyActivation} from '../utils/fused_utils';

import {add} from './Add';
import {batchMatMul} from './BatchMatMul';

export function _fusedMatMul(args: {
  inputs: _FusedMatMulInputs,
  attrs: _FusedMatMulAttrs,
  backend: MathBackendCPU
}) {
  const {inputs, backend, attrs} = args;
  const {a, b, bias, preluActivationWeights} = inputs;
  const {transposeA, transposeB, activation, leakyreluAlpha} = attrs;

  let current;
  let addRes;
  let activationRes;

  const intermediates: TensorInfo[] = [];

  const matMulRes =
      batchMatMul({inputs: {a, b}, attrs: {transposeA, transposeB}, backend});
  current = matMulRes;

  if (bias) {
    addRes = add({inputs: {a: current, b: bias}, backend}) as TensorInfo;
    intermediates.push(current);
    current = addRes;
  }
  if (activation) {
    activationRes = applyActivation(
        backend, current, activation, preluActivationWeights, leakyreluAlpha);
    intermediates.push(current);
    current = activationRes;
  }

  for (const i of intermediates) {
    backend.disposeIntermediateTensorInfo(i);
  }

  return current;
}

export const _fusedMatMulConfig: KernelConfig = {
  kernelName: _FusedMatMul,
  backendName: 'cpu',
  kernelFunc: _fusedMatMul as {} as KernelFunc,
};
