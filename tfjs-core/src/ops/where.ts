/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {SelectV2, SelectV2Inputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, assertShapesMatch} from '../util';

import {op} from './operation';
import {zerosLike} from './tensor_ops';

/**
 * Returns the elements, either `a` or `b` depending on the `condition`.
 *
 * If the condition is true, select from `a`, otherwise select from `b`.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const a = tf.tensor1d([1 , 2, 3]);
 * const b = tf.tensor1d([-1, -2, -3]);
 *
 * a.where(cond, b).print();
 * ```
 *
 * @param condition The input condition. Must be of dtype bool.
 * @param a If `condition` is rank 1, `a` may have a higher rank but
 *     its first dimension must match the size of `condition`.
 * @param b A tensor with the same shape and type as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function where_<T extends Tensor>(
    condition: Tensor|TensorLike, a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'where');
  const $b = convertToTensor(b, 'b', 'where');
  const $condition = convertToTensor(condition, 'condition', 'where', 'bool');

  assertShapesMatch($a.shape, $b.shape, 'Error in where: ');

  if ($condition.rank === 1) {
    // If condition rank is 1, then the first dimension must match the size of
    // condition.
    assert(
        $condition.shape[0] === $a.shape[0],
        () => 'The first dimension of `a` must match the size of `condition`.');
  } else {
    // A must have the same shape as condition.
    assertShapesMatch($condition.shape, $b.shape, 'Error in where: ');
  }

  // TODO(julianoks): Return null for condition gradient
  // when backprop supports it.
  const grad = (dy: T, saved: Tensor[]) => {
    const [$condition] = saved;
    return {
      condition: () => zerosLike($condition).toFloat(),
      t: () => dy.mul($condition.cast(dy.dtype)),
      e: () => dy.mul($condition.logicalNot().cast(dy.dtype))
    } as {t: () => T, e: () => T, condition: () => T};
  };

  const inputs: SelectV2Inputs = {condition: $condition, t: $a, e: $b};
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.select($condition, $a, $b);
    save([$condition]);
    return res;
  }, inputs as unknown as NamedTensorMap, grad, SelectV2) as T;
}

export const where = op({where_});
