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
import {Tensor1D, Tensor2D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {reshape} from './array_ops';
import {matMul} from './mat_mul';
import {op} from './operation';

/**
 * Computes the outer product of two vectors, `v1` and `v2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([3, 4, 5]);
 *
 * tf.outerProduct(a, b).print();
 * ```
 * @param v1 The first vector in the outer product operation.
 * @param v2 The second vector in the outer product operation.
 */
/** @doc {heading: 'Operations', subheading: 'Matrices'} */
function outerProduct_(
    v1: Tensor1D|TensorLike, v2: Tensor1D|TensorLike): Tensor2D {
  const $v1 = convertToTensor(v1, 'v1', 'outerProduct');
  const $v2 = convertToTensor(v2, 'v2', 'outerProduct');

  util.assert(
      $v1.rank === 1 && $v2.rank === 1,
      () => `Error in outerProduct: inputs must be rank 1, but got ranks ` +
          `${$v1.rank} and ${$v2.rank}.`);

  const v12D = reshape($v1, [-1, 1]);
  const v22D = reshape($v2, [1, -1]);
  return matMul(v12D, v22D) as Tensor2D;
}

export const outerProduct = op({outerProduct_});
