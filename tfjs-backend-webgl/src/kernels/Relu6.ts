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

import {KernelConfig, KernelFunc, Relu6} from '@tensorflow/tfjs-core';
import {unaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {CHECK_NAN_SNIPPET} from '../unaryop_gpu';

const RELU6 = CHECK_NAN_SNIPPET + `
  return (x < 0.0) ? 0.0 : min(6.0, x);
`;

const RELU6_PACKED = `
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;

export const relu6 =
    unaryKernelFunc({opSnippet: RELU6, packedOpSnippet: RELU6_PACKED});

export const relu6Config: KernelConfig = {
  kernelName: Relu6,
  backendName: 'webgl',
  kernelFunc: relu6 as {} as KernelFunc
};
