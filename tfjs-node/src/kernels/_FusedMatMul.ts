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

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, add, KernelConfig, matMul, Tensor, Tensor3D, tidy} from '@tensorflow/tfjs';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const _fusedMatMulConfig: KernelConfig = {
  kernelName: _FusedMatMul,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {a, b, bias, preluActivationWeights} =
        args.inputs as _FusedMatMulInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {transposeA, transposeB, activation, leakyreluAlpha} =
        args.attrs as unknown as _FusedMatMulAttrs;

    // Core TensorFlow does not have a fused BatchMatMul op. Combine calls to
    // achieve the same results:
    return tidy(() => {
      let result: Tensor3D =
          matMul(a as Tensor, b as Tensor, transposeA, transposeB);
      if (bias != null) {
        result = add(result, bias as Tensor);
      }

      result = backend.applyActivation(
          result, activation, preluActivationWeights as Tensor, leakyreluAlpha);

      return result;
    });
  }
};
