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
import {pool} from '../../ops/pool';
import {Tensor, Tensor3D, Tensor4D} from '../../tensor';
import {Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    pool<T extends Tensor3D|Tensor4D>(
        windowShape: [number, number]|number, poolingType: 'avg'|'max',
        padding: 'valid'|'same'|number, diationRate?: [number, number]|number,
        strides?: [number, number]|number): T;
  }
}

Tensor.prototype.pool = function<T extends Tensor3D|Tensor4D>(
    this: T, windowShape: [number, number]|number, poolingType: 'max'|'avg',
    padding: 'valid'|'same'|number, dilationRate?: [number, number]|number,
    strides?: [number, number]|number): T {
  this.throwIfDisposed();
  return pool(this, windowShape, poolingType, padding, dilationRate, strides);
};
