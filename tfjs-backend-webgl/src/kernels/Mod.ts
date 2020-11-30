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

import {KernelConfig, KernelFunc, Mod} from '@tensorflow/tfjs-core';

import {CHECK_NAN_SNIPPET} from '../binaryop_packed_gpu';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const MOD = `if (b == 0.0) return NAN;
  return mod(a, b);`;

const MOD_PACKED = `
  vec4 result = mod(a, b);
  vec4 isNaN = vec4(equal(b, vec4(0.0)));
  ` +
    CHECK_NAN_SNIPPET + `
  return result;
`;

export const mod = binaryKernelFunc({
  opSnippet: MOD,
  packedOpSnippet: MOD_PACKED,
});

export const modConfig: KernelConfig = {
  kernelName: Mod,
  backendName: 'webgl',
  kernelFunc: mod as {} as KernelFunc
};
