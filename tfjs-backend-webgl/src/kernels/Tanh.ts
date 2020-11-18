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

import {KernelConfig, Tanh} from '@tensorflow/tfjs-core';
import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const TANH = `
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`;

export const tanh = unaryKernelFunc({opSnippet: TANH});

export const tanhConfig: KernelConfig = {
  kernelName: Tanh,
  backendName: 'webgl',
  kernelFunc: tanh,
};
