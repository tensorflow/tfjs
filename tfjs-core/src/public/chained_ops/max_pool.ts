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
import {maxPool} from '../../ops/max_pool';
import {getGlobalTensorClass, Tensor3D, Tensor4D} from '../../tensor';
import {Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    maxPool<T extends Tensor3D|Tensor4D>(
        filterSize: [number, number]|number, strides: [number, number]|number,
        pad: 'valid'|'same'|number,
        dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  }
}

getGlobalTensorClass().prototype.maxPool =
    function<T extends Tensor3D|Tensor4D>(
        this: T, filterSize: [number, number]|number,
        strides: [number, number]|number, pad: 'valid'|'same'|number,
        dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  this.throwIfDisposed();
  return maxPool(this, filterSize, strides, pad, dimRoundingMode);
};
