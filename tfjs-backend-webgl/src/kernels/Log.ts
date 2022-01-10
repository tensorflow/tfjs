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

import {KernelConfig, KernelFunc, Log} from '@tensorflow/tfjs-core';

import {CHECK_NAN_SNIPPET_UNARY, unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {logImplCPU} from '../kernel_utils/shared';

const LOG = CHECK_NAN_SNIPPET_UNARY + `return log(x);`;

const LOG_PACKED = `
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
    result = vec4(NAN);
  }
  return result;
`;

export const log = unaryKernelFunc(
    {opSnippet: LOG, packedOpSnippet: LOG_PACKED, cpuKernelImpl: logImplCPU});

export const logConfig: KernelConfig = {
  kernelName: Log,
  backendName: 'webgl',
  kernelFunc: log as {} as KernelFunc
};
