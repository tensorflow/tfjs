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
import {batchNorm} from '../../ops/batchnorm';
import {getGlobalTensorClass, Tensor, Tensor1D} from '../../tensor';
import {Rank, TensorLike} from '../../types';

declare module '../../tensor' {
  interface Tensor<R extends Rank = Rank> {
    batchNorm<T extends Tensor>(
        mean: Tensor<R>|Tensor1D|TensorLike,
        variance: Tensor<R>|Tensor1D|TensorLike,
        offset?: Tensor<R>|Tensor1D|TensorLike,
        scale?: Tensor<R>|Tensor1D|TensorLike,
        varianceEpsilon?: number): Tensor<R>;
  }
}

getGlobalTensorClass().prototype.batchNorm = function<R extends Rank>(
    mean: Tensor<R>|Tensor1D|TensorLike,
    variance: Tensor<R>|Tensor1D|TensorLike,
    offset?: Tensor<R>|Tensor1D|TensorLike,
    scale?: Tensor<R>|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor<R> {
  this.throwIfDisposed();
  return batchNorm(this, mean, variance, offset, scale, varianceEpsilon);
};
