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

import {KernelConfig, Softplus} from '@tensorflow/tfjs-core';

import {unaryKernelFunc} from '../utils/unary_utils';

// mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX

// epsilon is the difference between 1.0 and the next representable float.
// For a single precision 32 bit float this should be 2^-23, see:
// https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
const epsilon = 1.1920928955078125e-7;
const threshold = Math.log(epsilon) + 2.0;

export const softplusKernelFunc = unaryKernelFunc(Softplus, (xi) => {
  // Value above which exp(x) may overflow, but softplus(x) == x
  // is within machine epsilon.
  const tooLarge = xi > -threshold;

  // Value below which exp(x) may underflow, but softplus(x) == exp(x)
  // is within machine epsilon.
  const tooSmall = xi < threshold;

  const expX = Math.exp(xi);
  let result;

  if (tooSmall) {
    result = expX;
  } else if (tooLarge) {
    result = xi;
  } else {
    result = Math.log(1.0 + expX);
  }
  return result;
});

export const softplusConfig: KernelConfig = {
  kernelName: Softplus,
  backendName: 'cpu',
  kernelFunc: softplusKernelFunc,
};
