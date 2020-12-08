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
import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {logImplCPU} from '../kernel_utils/shared';

const LOG = `if (x < 0.0) return NAN;
  return log(x);`;

const LOG_PACKED = `
  vec4 result = log(x);
  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));
  result.r = isNaN.r == 1.0 ? NAN : result.r;
  result.g = isNaN.g == 1.0 ? NAN : result.g;
  result.b = isNaN.b == 1.0 ? NAN : result.b;
  result.a = isNaN.a == 1.0 ? NAN : result.a;

  return result;
`;

export const log = unaryKernelFunc(
    {opSnippet: LOG, packedOpSnippet: LOG_PACKED, cpuKernelImpl: logImplCPU});

export const logConfig: KernelConfig = {
  kernelName: Log,
  backendName: 'webgl',
  kernelFunc: log as {} as KernelFunc
};
