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

import {reshape} from '../../ops/reshape';
import {getGlobalTensorClass, Tensor} from '../../tensor';
import {Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    as4D<T extends Tensor>(
        rows: number, columns: number, depth: number, depth2: number): Tensor4D;
  }
}

/**
 * Converts a `tf.Tensor` to a `tf.Tensor4D`.
 *
 * @param rows Number of rows in `tf.Tensor4D`.
 * @param columns Number of columns in `tf.Tensor4D`.
 * @param depth Depth of `tf.Tensor4D`.
 * @param depth2 4th dimension of `tf.Tensor4D`.
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
getGlobalTensorClass().prototype.as4D = function<T extends Tensor>(
    rows: number, columns: number, depth: number, depth2: number): T {
  this.throwIfDisposed();
  return reshape(this, [rows, columns, depth, depth2]) as T;
};
