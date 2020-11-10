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
import {dilation2d} from '../../ops/dilation2d';
import {Tensor, Tensor3D, Tensor4D} from '../../tensor';
import {Rank, TensorLike3D} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    dilation2d<T extends Tensor3D|Tensor4D>(
        filter: Tensor3D|TensorLike3D, strides: [number, number]|number,
        pad: 'valid'|'same', dilations?: [number, number]|number,
        dataFormat?: 'NHWC'): T;
  }
}

Tensor.prototype.dilation2d = function<T extends Tensor3D|Tensor4D>(
    filter: Tensor3D|TensorLike3D, strides: [number, number]|number,
    pad: 'valid'|'same', dilations?: [number, number]|number,
    dataFormat?: 'NHWC'): T {
  this.throwIfDisposed();
  return dilation2d(this, filter, strides, pad, dilations, dataFormat) as T;
};
