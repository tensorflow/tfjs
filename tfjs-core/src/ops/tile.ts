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
import {Tile, TileAttrs, TileInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {DataType, TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Construct a tensor by repeating it the number of times given by reps.
 *
 * This operation creates a new tensor by replicating `input` `reps`
 * times. The output tensor's i'th dimension has `input.shape[i] *
 * reps[i]` elements, and the values of `input` are replicated
 * `reps[i]` times along the i'th dimension. For example, tiling
 * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 *
 * a.tile([2]).print();    // or a.tile([2])
 * ```
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.tile([1, 2]).print();  // or a.tile([1, 2])
 * ```
 * @param x The tensor to tile.
 * @param reps Determines the number of replications per dimension.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function tile_<T extends Tensor>(x: T|TensorLike, reps: number[]): T {
  const parseAs: DataType = null;
  const $x = convertToTensor(x, 'x', 'tile', parseAs);
  util.assert(
      $x.rank === reps.length,
      () => `Error in transpose: rank of input ${$x.rank} ` +
          `must match length of reps ${reps}.`);

  const forward: ForwardFunc<T> = (backend, save) => {
    const res = backend.tile($x, reps);
    save([$x]);
    return res;
  };

  const inputsToSave = [$x];
  const inputs: TileInputs = {x: $x};
  const attrs: TileAttrs = {reps};

  return ENGINE.runKernelFunc(
      forward, inputs as unknown as NamedTensorMap, null /* grad */, Tile,
      attrs as unknown as NamedAttrMap, inputsToSave);
}

export const tile = op({tile_});
