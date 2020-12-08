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
import {ArgMin, ArgMinAttrs, ArgMinInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Returns the indices of the minimum values along an `axis`.
 *
 * The result has the same shape as `input` with the dimension along `axis`
 * removed.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.argMin().print();  // or tf.argMin(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
 *
 * const axis = 1;
 * x.argMin(axis).print();  // or tf.argMin(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function argMin_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  const $x = convertToTensor(x, 'x', 'argMin');

  const inputs: ArgMinInputs = {x: $x};
  const attrs: ArgMinAttrs = {axis};

  return ENGINE.runKernel(
      ArgMin, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const argMin = op({argMin_});
