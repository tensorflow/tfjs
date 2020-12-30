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
import {ENGINE} from '../engine';
import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes the dot product of two matrices, A * B. These must be matrices.
 *
 * ```js
 * const a = tf.tensor2d([1, 2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.matMul(b).print();  // or tf.matMul(a, b)
 * ```
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function matMul_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike, transposeA = false,
    transposeB = false): T {
  let $a = convertToTensor(a, 'a', 'matMul');
  let $b = convertToTensor(b, 'b', 'matMul');
  [$a, $b] = makeTypesMatch($a, $b);

  const inputs: BatchMatMulInputs = {a: $a, b: $b};
  const attrs: BatchMatMulAttrs = {transposeA, transposeB};

  return ENGINE.runKernel(
      BatchMatMul, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const matMul = op({matMul_});
