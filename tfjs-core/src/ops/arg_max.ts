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

import {ENGINE, ForwardFunc} from '../engine';
import {ArgMax, ArgMaxAttrs, ArgMaxInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as axis_util from './axis_util';
import {op} from './operation';
import {transpose} from './transpose';

/**
 * Returns the indices of the maximum values along an `axis`.
 *
 * The result has the same shape as `input` with the dimension along `axis`
 * removed.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.argMax().print();  // or tf.argMax(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
 *
 * const axis = 1;
 * x.argMax(axis).print();  // or tf.argMax(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function argMax_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  let $x = convertToTensor(x, 'x', 'argMax');

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    save([$x]);

    let axes = util.parseAxisParam(axis, $x.shape);
    const permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
      $x = transpose($x, permutedAxes);
      axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    return backend.argMax($x, axes[0]);
  };

  const inputs: ArgMaxInputs = {x: $x};
  const attrs: ArgMaxAttrs = {axis};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, ArgMax,
             attrs as {} as NamedAttrMap) as T;
}

export const argMax = op({argMax_});
