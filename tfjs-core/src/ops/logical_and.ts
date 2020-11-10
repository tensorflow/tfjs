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
import {LogicalAnd, LogicalAndInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {assertAndGetBroadcastShape} from './broadcast_util';
import {op} from './operation';

/**
 * Returns the truth value of `a AND b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalAnd(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalAnd_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'logicalAnd', 'bool');
  const $b = convertToTensor(b, 'b', 'logicalAnd', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);

  const inputs: LogicalAndInputs = {a: $a, b: $b};

  return ENGINE.runKernelFunc(
             backend => backend.logicalAnd($a, $b),
             inputs as {} as NamedTensorMap, null /* grad */, LogicalAnd) as T;
}

export const logicalAnd = op({logicalAnd_});
