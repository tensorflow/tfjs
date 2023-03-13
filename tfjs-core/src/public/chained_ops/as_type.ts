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

// TODO update import path once op is modularized.
import {cast} from '../../ops/ops';
import {getGlobalTensorClass, Tensor} from '../../tensor';
import {DataType, Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    asType<T extends Tensor>(this: T, dtype: DataType): T;
  }
}

/**
 * Casts a `tf.Tensor` to a specified dtype.
 *
 * @param dtype Data-type to cast the tensor to.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
getGlobalTensorClass().prototype.asType = function<T extends Tensor>(
    this: T, dtype: DataType): T {
  this.throwIfDisposed();
  return cast<T>(this, dtype);
};
