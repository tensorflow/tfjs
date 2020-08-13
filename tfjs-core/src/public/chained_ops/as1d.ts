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
import {Tensor} from '../../tensor';
import {Rank} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    as1D<T extends Tensor>(): Tensor1D;
  }
}

/** Converts a `tf.Tensor` to a `tf.Tensor1D`. */
/** @doc {heading: 'Tensors', subheading: 'Classes'} */
Tensor.prototype.as1D = function<T extends Tensor>(): T {
  this.throwIfDisposed();
  return reshape(this, [this.size]) as T;
};
