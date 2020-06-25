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
import {LogicalNot, LogicalNotInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * Returns the truth value of `NOT x` element-wise.
 *
 * ```js
 * const a = tf.tensor1d([false, true], 'bool');
 *
 * a.logicalNot().print();
 * ```
 *
 * @param x The input tensor. Must be of dtype 'bool'.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function logicalNot_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'logicalNot', 'bool');
  const inputs: LogicalNotInputs = {x: $x};
  return ENGINE.runKernelFunc(
      backend => backend.logicalNot($x), inputs as {} as NamedTensorMap,
      null /* grad */, LogicalNot);
}

export const logicalNot = op({logicalNot_});
