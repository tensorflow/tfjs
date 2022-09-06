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

import {KernelConfig, KernelFunc, LogicalOr} from '@tensorflow/tfjs-core';

import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const LOGICAL_OR = `return float(a >= 1.0 || b >= 1.0);`;
const LOGICAL_OR_PACKED = `
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`;

export const logicalOr = binaryKernelFunc(
    {opSnippet: LOGICAL_OR, packedOpSnippet: LOGICAL_OR_PACKED, dtype: 'bool'});

export const logicalOrConfig: KernelConfig = {
  kernelName: LogicalOr,
  backendName: 'webgl',
  kernelFunc: logicalOr as {} as KernelFunc
};
