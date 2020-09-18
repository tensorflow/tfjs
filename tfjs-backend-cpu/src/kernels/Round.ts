/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {KernelConfig, Round} from '@tensorflow/tfjs-core';

import {unaryKernelFunc} from '../utils/kernel_utils';

export const roundKernelFunc = unaryKernelFunc(Round, (x) => {
  // The algorithm is based on banker's rounding.
  const base = Math.floor(x);
  if (x - base < 0.5) {
    return Math.floor(x);
  } else if (x - base > 0.5) {
    return Math.ceil(x);
  } else {
    if (base % 2.0 === 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
});

export const roundConfig: KernelConfig = {
  kernelName: Round,
  backendName: 'cpu',
  kernelFunc: roundKernelFunc,
};
