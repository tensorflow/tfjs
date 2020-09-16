/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {ENGINE, ForwardFunc} from '../engine';
import {OnesLike, OnesLikeInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {complex} from './complex';
import {imag} from './imag';
import {op} from './operation';
import {real} from './real';
import {zerosLike} from './zeros_like';

/**
 * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.onesLike(x).print();
 * ```
 * @param x A tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function onesLike_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'onesLike');

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    if ($x.dtype === 'complex64') {
      const r = onesLike(real($x));
      const i = zerosLike(imag($x));
      return complex(r, i);
    }

    return backend.onesLike($x);
  };

  const inputs: OnesLikeInputs = {x: $x};
  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             OnesLike) as T;
}

export const onesLike = op({onesLike_});
