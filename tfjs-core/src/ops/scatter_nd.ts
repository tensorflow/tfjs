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
import {ScatterNd, ScatterNdAttrs, ScatterNdInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';

import {op} from './operation';
import * as scatter_nd_util from './scatter_nd_util';

/**
 * Creates a new tensor by applying sparse updates to individual
 * values or slices within a zero tensor of the given shape tensor according to
 * indices. This operator is the inverse of the `tf.gatherND` operator which
 * extracts values or slices from a given tensor.
 *
 * ```js
 * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
 * const updates = tf.tensor1d([9, 10, 11, 12]);
 * const shape = [8];
 * tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
 * ```
 *
 * @param indices The tensor contains the indices into the output tensor.
 * @param updates The tensor contains the value for the indices.
 * @param shape: The shape of the output tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function scatterND_<R extends Rank>(
    indices: Tensor|TensorLike, updates: Tensor|TensorLike,
    shape: ShapeMap[R]): Tensor<R> {
  const $indices = convertToTensor(indices, 'indices', 'scatterND', 'int32');
  const $updates = convertToTensor(updates, 'updates', 'scatterND');
  scatter_nd_util.validateInput($updates, $indices, shape);

  const forward: ForwardFunc<Tensor> = (backend) => {
    return backend.scatterND($indices, $updates, shape);
  };

  const inputs: ScatterNdInputs = {indices: $indices, updates: $updates};
  const attrs: ScatterNdAttrs = {shape};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null /* grad */,
             ScatterNd, attrs as {} as NamedAttrMap) as Tensor<R>;
}

export const scatterND = op({scatterND_});
