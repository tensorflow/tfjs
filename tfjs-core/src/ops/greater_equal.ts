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
import {ENGINE, ForwardFunc} from '../engine';
import {GreaterEqual, GreaterEqualInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {assertAndGetBroadcastShape} from './broadcast_util';
import {op} from './operation';

/**
 * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.greaterEqual(b).print();
 * ```
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function greaterEqual_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'greaterEqual');
  let $b = convertToTensor(b, 'b', 'greaterEqual');
  [$a, $b] = makeTypesMatch($a, $b);

  assertAndGetBroadcastShape($a.shape, $b.shape);

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const res = backend.greaterEqual($a, $b);
    save([$a, $b]);
    return res;
  };

  const inputs: GreaterEqualInputs = {a: $a, b: $b};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             GreaterEqual) as T;
}

export const greaterEqual = op({greaterEqual_});
