/**
 * @license
 * Copyright 2023 Google LLC.
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
import {BitwiseAnd, BitwiseAndInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {Rank} from '../types';
import {arraysEqual} from '../util_base';

import {op} from './operation';

/**
 * Bitwise `AND` operation for input tensors.
 *
 * Given two input tensors, returns a new tensor
 * with the `AND` calculated values.
 *
 * The method supports int32 values
 *
 *
 * ```js
 * const x = tf.tensor1d([0, 5, 3, 14], 'int32');
 * const y = tf.tensor1d([5, 0, 7, 11], 'int32');
 * tf.bitwiseAnd(x, y).print();
 * ```
 *
 * @param x The input tensor to be calculated.
 * @param y The input tensor to be calculated.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function bitwiseAnd_<R extends Rank>(x: Tensor, y: Tensor): Tensor<R> {
  const $x = convertToTensor(x, 'x', 'bitwiseAnd');
  const $y = convertToTensor(y, 'y', 'bitwiseAnd');

  if (!arraysEqual($x.shape, $y.shape)) {
    throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${
        $x.shape}, y: ${$y.shape}`);
  }
  if ($x.dtype !== 'int32' || $y.dtype !== 'int32') {
    throw new Error(
        `BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${
            $x.dtype} and type of y: ${$y.dtype}`);
  }

  const inputs: BitwiseAndInputs = {a: $x, b: $y};
  return ENGINE.runKernel(BitwiseAnd, inputs as unknown as NamedTensorMap);
}
export const bitwiseAnd = /* @__PURE__ */ op({bitwiseAnd_});
