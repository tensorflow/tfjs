/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {Tensor1D} from '../tensor';
import {tensor1d} from './tensor1d';

export function enclosingPowerOfTwo(value: number) {
  // Return 2**N for integer N such that 2**N >= value.
  return Math.floor(Math.pow(2, Math.ceil(Math.log(value) / Math.log(2.0))));
}

export function cosineWindow(
    windowLength: number, a: number, b: number): Tensor1D {
  const even = 1 - windowLength % 2;
  const newValues = new Float32Array(windowLength);
  for (let i = 0; i < windowLength; ++i) {
    const cosArg = (2.0 * Math.PI * i) / (windowLength + even - 1);
    newValues[i] = a - b * Math.cos(cosArg);
  }
  return tensor1d(newValues, 'float32');
}
