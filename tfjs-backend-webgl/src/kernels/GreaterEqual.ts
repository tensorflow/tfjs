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

import {GreaterEqual, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {greaterEqualImplCPU} from '../kernel_utils/shared';

const GREATER_EQUAL = `return float(a >= b);`;
const GREATER_EQUAL_PACKED = `
  return vec4(greaterThanEqual(a, b));
`;

export const greaterEqual = binaryKernelFunc({
  opSnippet: GREATER_EQUAL,
  packedOpSnippet: GREATER_EQUAL_PACKED,
  dtype: 'bool',
  cpuKernelImpl: greaterEqualImplCPU
});

export const greaterEqualConfig: KernelConfig = {
  kernelName: GreaterEqual,
  backendName: 'webgl',
  kernelFunc: greaterEqual as {} as KernelFunc
};
