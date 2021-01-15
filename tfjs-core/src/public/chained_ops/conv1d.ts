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
import {conv1d} from '../../ops/conv1d';
import {ExplicitPadding} from '../../ops/conv_util';
import {getGlobalTensorClass, Tensor2D, Tensor3D} from '../../tensor';
import {Rank, TensorLike3D} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    conv1d<T extends Tensor2D|Tensor3D>(
        filter: Tensor3D|TensorLike3D, stride: number,
        pad: 'valid'|'same'|number|ExplicitPadding, dataFormat?: 'NWC'|'NCW',
        dilation?: number, dimRoundingMode?: 'floor'|'round'|'ceil'): T;
  }
}

getGlobalTensorClass().prototype.conv1d = function<T extends Tensor2D|Tensor3D>(
    filter: Tensor3D|TensorLike3D, stride: number,
    pad: 'valid'|'same'|number|ExplicitPadding, dataFormat?: 'NWC'|'NCW',
    dilation?: number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  this.throwIfDisposed();
  return conv1d(
             this, filter, stride, pad, dataFormat, dilation,
             dimRoundingMode) as T;
};
