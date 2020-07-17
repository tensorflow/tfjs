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
import {Concat, ConcatAttrs, ConcatInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensorArray} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, parseAxisParam, sizeFromShape} from '../util';

import {assertParamsConsistent, computeOutShape} from './concat_util';
import {op} from './operation';
import {tensor} from './tensor';

/**
 * Concatenates a list of `tf.Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concate along. Defaults to 0 (the first dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function concat_<T extends Tensor>(tensors: Array<T|TensorLike>, axis = 0): T {
  assert(tensors.length >= 1, () => 'Pass at least one tensor to concat');

  let $tensors = convertToTensorArray(tensors, 'tensors', 'concat');
  if ($tensors[0].dtype === 'complex64') {
    $tensors.forEach(tensor => {
      if (tensor.dtype !== 'complex64') {
        throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${tensor.dtype}. `);
      }
    });
  }

  const $axis = parseAxisParam(axis, $tensors[0].shape)[0];
  const outShape = computeOutShape($tensors.map(t => t.shape), $axis);
  if (sizeFromShape(outShape) === 0) {
    return tensor([], outShape) as T;
  }
  // Keep only non-empty tensors (ignore tensors with 0 in their shape).
  $tensors = $tensors.filter(t => t.size > 0);
  if ($tensors.length === 1) {
    return $tensors[0];
  }

  const shapes = $tensors.map(t => t.shape);
  assertParamsConsistent(shapes, $axis);

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const res = backend.concat($tensors, $axis);
    save($tensors);
    return res;
  };

  const inputs: ConcatInputs = $tensors;
  const attr: ConcatAttrs = {axis};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */, Concat,
             attr as {} as NamedAttrMap) as T;
}

export const concat = op({concat_});
