/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {Reverse, ReverseAttrs, ReverseInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {parseAxisParam} from '../util';

import {clone} from './clone';
import {op} from './operation';
import {reshape} from './reshape';

/**
 * Reverses a `tf.Tensor` along a specified axis.
 *
 * Also available are stricter rank-specific methods that assert that `x` is
 * of the given rank:
 *   - `tf.reverse1d`
 *   - `tf.reverse2d`
 *   - `tf.reverse3d`
 *   - `tf.reverse4d`
 *
 * Except `tf.reverse1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.reverse().print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.reverse(axis).print();
 * ```
 * @param x The input tensor to be reversed.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function reverse_<T extends Tensor>(
    x: T|TensorLike, axis?: number|number[]): T {
  const $x = convertToTensor(x, 'x', 'reverse');

  const forward: ForwardFunc<Tensor> = (backend) => {
    const axes = parseAxisParam(axis, $x.shape);
    if ($x.rank === 0) {
      return clone($x);
    }
    const res = backend.reverse($x, axes);
    return reshape(res, $x.shape);
  };

  const inputs: ReverseInputs = {x: $x};
  const attrs: ReverseAttrs = {dims: axis};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* gradient */,
             Reverse, attrs as {} as NamedAttrMap) as T;
}

export const reverse = op({reverse_});
