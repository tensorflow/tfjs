/**
 * @license
 * Copyright 2023 Google LLC.
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

import {BitwiseAnd, KernelConfig} from '@tensorflow/tfjs-core';

import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {bitwiseAndImplCPU as cpuBitwiseAnd} from '../kernel_utils/shared';

const BITWISEAND = 'return a & b;';

export const addKernelFunc = binaryKernelFunc({
  opSnippet: BITWISEAND,
  packedOpSnippet: BITWISEAND,
  supportsComplex: true,
  cpuKernelImpl: cpuBitwiseAnd
});

export const addConfig: KernelConfig = {
  kernelName: BitwiseAnd,
  backendName: 'webgl',
  kernelFunc: addKernelFunc
};
