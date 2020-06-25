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

import {Tensor} from '../tensor';
import {convertToTensorArray} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {concat} from './concat';
import {expandDims} from './expand_dims';
import {op} from './operation';

/**
 * Stacks a list of rank-`R` `tf.Tensor`s into one rank-`(R+1)` `tf.Tensor`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.stack([a, b, c]).print();
 * ```
 *
 * @param tensors A list of tensor objects with the same shape and dtype.
 * @param axis The axis to stack along. Defaults to 0 (the first dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function stack_<T extends Tensor>(
    tensors: Array<T|TensorLike>, axis = 0): Tensor {
  const $tensors = convertToTensorArray(tensors, 'tensors', 'stack');

  util.assert(
      $tensors.length >= 1, () => 'Pass at least one tensor to tf.stack');

  if ($tensors.length === 1) {
    return expandDims($tensors[0], axis);
  }

  const rank = $tensors[0].rank;
  const shape = $tensors[0].shape;
  const dtype = $tensors[0].dtype;

  util.assert(axis <= rank, () => 'Axis must be <= rank of the tensor');

  $tensors.forEach(t => {
    util.assertShapesMatch(
        shape, t.shape,
        'All tensors passed to stack must have matching shapes');
    util.assert(
        dtype === t.dtype,
        () => 'All tensors passed to stack must have matching dtypes');
  });

  const expandedTensors = $tensors.map(t => expandDims(t, axis));
  // Stack exists in the TensorFlow C++ API
  // (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/stack) but not
  // in
  // https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/ops/ops.pbtxt.
  // Therefore we are treating it like a high-level op rather than
  // creating a dedicated stack kernel.
  return concat(expandedTensors, axis);
}

export const stack = op({stack_});
