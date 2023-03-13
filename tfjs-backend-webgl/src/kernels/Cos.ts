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

import {Cos, KernelConfig} from '@tensorflow/tfjs-core';

import {CHECK_NAN_SNIPPET_PACKED} from '../binaryop_packed_gpu';
import {CHECK_NAN_SNIPPET_UNARY, unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const COS = CHECK_NAN_SNIPPET_UNARY + `
  return cos(x);
`;

const COS_PACKED = `
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${CHECK_NAN_SNIPPET_PACKED}
  return result;
`;

export const cos =
    unaryKernelFunc({opSnippet: COS, packedOpSnippet: COS_PACKED});

export const cosConfig: KernelConfig = {
  kernelName: Cos,
  backendName: 'webgl',
  kernelFunc: cos,
};
