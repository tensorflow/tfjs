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

import {Expm1, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {expm1ImplCPU} from '../kernel_utils/shared';

const EXPM1 = `return exp(x) - 1.0;`;

export const expm1 = unaryKernelFunc(
    {opSnippet: EXPM1, packedOpSnippet: EXPM1, cpuKernelImpl: expm1ImplCPU});

export const expm1Config: KernelConfig = {
  kernelName: Expm1,
  backendName: 'webgl',
  kernelFunc: expm1 as {} as KernelFunc
};
