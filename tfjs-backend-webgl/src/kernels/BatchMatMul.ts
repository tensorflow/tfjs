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

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, env, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {batchMatMulImpl, batchMatMulMrt2x2Impl} from './BatchMatMul_impl';

export function batchMatMul(args: {
  inputs: BatchMatMulInputs,
  attrs: BatchMatMulAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {a, b} = inputs;
  const {transposeA, transposeB} = attrs;

  if (env().getBool('WEBGL2_USE_MRT_FOR_MATMUL')) {
    return batchMatMulMrt2x2Impl({a, b, transposeA, transposeB, backend});
  }

  return batchMatMulImpl({a, b, transposeA, transposeB, backend});
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'webgl',
  kernelFunc: batchMatMul as {} as KernelFunc,
};
