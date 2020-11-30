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

import {Equal, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const PACKED_EQUAL = `
  return vec4(equal(a, b));
`;

const EQUAL = `return float(a == b);`;

export const equal = binaryKernelFunc(
    {opSnippet: EQUAL, packedOpSnippet: PACKED_EQUAL, dtype: 'bool'});

export const equalConfig: KernelConfig = {
  kernelName: Equal,
  backendName: 'webgl',
  kernelFunc: equal as {} as KernelFunc
};
