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

import {backend_util, Erf, KernelConfig} from '@tensorflow/tfjs-core';

import {unaryKernelFunc} from '../utils/unary_utils';

const p = backend_util.ERF_P;
const a1 = backend_util.ERF_A1;
const a2 = backend_util.ERF_A2;
const a3 = backend_util.ERF_A3;
const a4 = backend_util.ERF_A4;
const a5 = backend_util.ERF_A5;

export const erf = unaryKernelFunc(
    Erf,
    (xi) => {
      const sign = Math.sign(xi);
      const v = Math.abs(xi);
      const t = 1.0 / (1.0 + p * v);
      return sign *
          (1.0 -
           (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
               Math.exp(-v * v));
    },
);

export const erfConfig: KernelConfig = {
  kernelName: Erf,
  backendName: 'cpu',
  kernelFunc: erf,
};
