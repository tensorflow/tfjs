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

import {KernelConfig, KernelFunc, Minimum} from '@tensorflow/tfjs-core';

import {CHECK_NAN_SNIPPET} from '../binaryop_gpu';
import {CHECK_NAN_SNIPPET as CHECK_NAN_SNIPPET_PACKED} from '../binaryop_packed_gpu';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {minimumImplCPU} from '../kernel_utils/shared';

const MINIMUM = CHECK_NAN_SNIPPET + `
  return min(a, b);
`;

const MINIMUM_PACKED = `
  vec4 result = vec4(min(a, b));
  vec4 isNaN = min(vec4(isnan(a)) + vec4(isnan(b)), vec4(1.0));
  ` +
    CHECK_NAN_SNIPPET_PACKED + `
  return result;
`;

export const minimum = binaryKernelFunc({
  opSnippet: MINIMUM,
  packedOpSnippet: MINIMUM_PACKED,
  cpuKernelImpl: minimumImplCPU
});

export const minimumConfig: KernelConfig = {
  kernelName: Minimum,
  backendName: 'webgl',
  kernelFunc: minimum as {} as KernelFunc
};
