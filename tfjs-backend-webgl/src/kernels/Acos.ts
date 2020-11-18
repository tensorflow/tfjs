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

import {Acos, KernelConfig} from '@tensorflow/tfjs-core';

import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {CHECK_NAN_SNIPPET} from '../unaryop_gpu';

const ACOS = CHECK_NAN_SNIPPET + `
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`;

export const acos = unaryKernelFunc({opSnippet: ACOS});

export const acosConfig: KernelConfig = {
  kernelName: Acos,
  backendName: 'webgl',
  kernelFunc: acos,
};
