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

import {ENGINE} from '../engine';
import {Tensor, TensorBuffer} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {DataType, DataTypeMap, Rank, ShapeMap, TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 * If one component of shape is the special value -1, the size of that
 * dimension is computed so that the total size remains constant. In
 * particular, a shape of [-1] flattens into 1-D. At most one component of
 * shape can be -1.
 *
 * If shape is 1-D or higher, then the operation returns a tensor with shape
 * shape filled with the values of tensor. In this case, the number of
 * elements implied by shape must be the same as the number of elements in
 * tensor.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.reshape([2, 2]).print();
 * ```
 *
 * @param x The input tensor to be reshaped.
 * @param shape An array of integers defining the output tensor shape.
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
function reshape_<R2 extends Rank>(
    x: Tensor|TensorLike, shape: ShapeMap[R2]): Tensor<R2> {
  const $x = convertToTensor(x, 'x', 'reshape', null);
  shape = util.inferFromImplicitShape(shape, $x.size) as ShapeMap[R2];
  util.assert(
      $x.size === util.sizeFromShape(shape),
      () => 'new shape and old shape must have the same number of elements.');

  const grad = (dy: Tensor<R2>) => {
    return {x: () => reshape(dy, $x.shape)};
  };
  const attrs = {shape};
  return ENGINE.runKernelFunc(
      backend => backend.reshape($x, shape), {x: $x}, grad, 'Reshape', attrs);
}
export const reshape = op({reshape_});
