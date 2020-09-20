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

import {ENGINE, ForwardFunc} from '../engine';
import {MirrorPad, MirrorPadAttrs, MirrorPadInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Pads a `tf.Tensor` using mirror padding.
 *
 * This operation implements the `REFLECT` and `SYMMETRIC` modes of pad.
 *
 * ```js
 * const x = tf.range(0, 9).reshape([1, 1, 3, 3]);
 * x.mirrorPad([[0, 0], [0, 0], [2, 2], [2, 2]]).print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * If `mode` is "reflect" then both `paddings[D, 0]` and `paddings[D, 1]`
 * must be no greater than `x.shape[D] - 1`. If mode is "symmetric"
 * then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than
 * `x.shape[D]`
 * @param mode Optional string from `'reflect' | 'symmetric'`,
 *     defaults to reflect, which specifies the padding mode
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function mirrorPad_<T extends Tensor>(
    x: T|TensorLike, paddings: Array<[number, number]>,
    mode?: 'reflect'|'symmetric'): T {
  mode = mode || 'reflect';
  const $x = convertToTensor(x, 'x', 'mirrorPad');
  if ($x.rank === 0) {
    throw new Error(
        'mirrorPad(scalar) is not defined. ' +
        'Pass non-scalar to mirrorPad');
  }
  util.assert(
      paddings.length === $x.rank,
      () => `Padding doesn't match input. Must be ${$x.rank}. ` +
          `Got ${paddings.length}.`);
  const offset = mode === 'reflect' ? 0 : 1;
  for (let i = 0; i < $x.rank; i++) {
    util.assert(
        paddings[i].length === 2,
        () => `Invalid number of paddings. Must be length of 2 each.`);
    util.assert(
        paddings[i][0] < $x.shape[i] + offset &&
            paddings[i][1] < $x.shape[i] + offset,
        () => `Padding in dimension ${i} cannot be greater than or equal ` +
            `to ${$x.shape[i] + offset} for input of shape ${$x.shape}`);
  }

  const forward: ForwardFunc<T> = (backend, save) => {
    save([$x]);
    return backend.mirrorPad($x, paddings, mode);
  };

  const attrs: MirrorPadAttrs = {paddings, mode};
  const inputs: MirrorPadInputs = {x: $x};
  return ENGINE.runKernelFunc(
      forward, inputs as unknown as NamedTensorMap, null /* grad */, MirrorPad,
      attrs as unknown as NamedAttrMap);
}

export const mirrorPad = op({mirrorPad_});
