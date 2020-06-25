/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * Computes square of `x` element-wise: `x ^ 2`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);
 *
 * x.square().print();  // or tf.square(x)
 * ```
 * @param x The input Tensor.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function square_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'square');
  const attrs = {};
  const inputsToSave = [$x];
  const outputsToSave: boolean[] = [];
  return ENGINE.runKernelFunc((backend, save) => {
    save([$x]);
    return backend.square($x);
  }, {x: $x}, null /* grad */, 'Square', attrs, inputsToSave, outputsToSave);
}

export const square = op({square_});
