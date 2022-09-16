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

import {Atan2} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';
import {CHECK_NAN_SNIPPET} from '../binaryop_gpu';
import {CHECK_NAN_SNIPPET_PACKED} from '../binaryop_packed_gpu';
import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';

const ATAN2 = CHECK_NAN_SNIPPET + `
  return atan(a, b);
`;

const ATAN2_PACKED = `
  vec4 result = atan(a, b);
  vec4 nanValue = vec4(NAN);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  ` +
    CHECK_NAN_SNIPPET_PACKED + `
  return result;
`;

export const atan2 =
    binaryKernelFunc({opSnippet: ATAN2, packedOpSnippet: ATAN2_PACKED});

export const atan2Config: KernelConfig = {
  kernelName: Atan2,
  backendName: 'webgl',
  kernelFunc: atan2,
};
