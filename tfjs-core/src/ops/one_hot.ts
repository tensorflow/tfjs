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
import {OneHot, OneHotAttrs, OneHotInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';
import {reshape} from './reshape';

/**
 * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
 * value `onValue` (defaults to 1), while all other locations take value
 * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
 * `R+1` with the last axis of size `depth`.
 *
 * ```js
 * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
 * ```
 *
 * @param indices `tf.Tensor` of indices with dtype `int32`.
 * @param depth The depth of the one hot dimension.
 * @param onValue A number used to fill in the output when the index matches
 * the location.
 * @param offValue A number used to fill in the output when the index does
 *     not match the location.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function oneHot_(
    indices: Tensor|TensorLike, depth: number, onValue = 1,
    offValue = 0): Tensor {
  if (depth < 2) {
    throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
  }
  const $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
  const outShape = [...$indices.shape, depth];

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    save([$indices]);
    return reshape(
        backend.oneHot(
            reshape($indices, [$indices.size]), depth, onValue, offValue),
        outShape);
  };

  const inputs: OneHotInputs = {indices: $indices};
  const attrs: OneHotAttrs = {depth, onValue, offValue};

  return ENGINE.runKernelFunc(
      forward, inputs as unknown as NamedTensorMap, null /* grad */, OneHot,
      attrs as unknown as NamedAttrMap);
}

export const oneHot = op({oneHot_});
