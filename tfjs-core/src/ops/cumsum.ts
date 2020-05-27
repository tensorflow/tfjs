/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {KernelBackend} from '../backends/backend';
import {ENGINE, ForwardFunc} from '../engine';
import {Cumsum, CumsumAttrs, CumsumInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {GradSaveFunc, NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {getAxesPermutation, getInnerMostAxes} from './axis_util';
import {op} from './operation';
import {transpose} from './transpose';

/**
 * Computes the cumulative sum of a `tf.Tensor` along `axis`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4]);
 * x.cumsum().print();
 * ```
 * ```js
 * const x = tf.tensor([[1, 2], [3, 4]]);
 * x.cumsum().print();
 * ```
 *
 * @param x The input tensor to be summed.
 * @param axis The axis along which to sum. Optional. Defaults to 0.
 * @param exclusive Whether to perform exclusive cumulative sum. Optional.
 *     Defaults to false. If set to true then the sum of each tensor entry
 *     does not include its own value, but only the values previous to it
 *     along the specified axis.
 * @param reverse Whether to sum in the opposite direction. Optional.
 *     Defaults to false.
 */
/** @doc {heading: 'Operations', subheading: 'Scan'} */
function cumsum_<T extends Tensor>(
    x: Tensor|TensorLike, axis = 0, exclusive = false, reverse = false): T {
  const $x = convertToTensor(x, 'x', 'cumsum');

  const forward: ForwardFunc<Tensor> =
      (backend: KernelBackend, save: GradSaveFunc) => {
        const permutation = getAxesPermutation([axis], $x.rank);
        let permutedX = $x;
        if (permutation != null) {
          permutedX = transpose($x, permutation);
        }
        const permutedAxis = getInnerMostAxes(1, $x.rank)[0];
        let value = backend.cumsum(permutedX, permutedAxis, exclusive, reverse);
        save([$x]);

        if (permutation != null) {
          value = transpose(value, permutation);
        }
        return value;
      };

  const inputs: CumsumInputs = {x: $x};
  const attrs: CumsumAttrs = {axis, exclusive, reverse};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Cumsum,
             attrs as {} as NamedAttrMap) as T;
}

export const cumsum = op({cumsum_});
