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
import {ExpandDims, ExpandDimsAttrs, ExpandDimsInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
 * into the tensor's shape.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const axis = 1;
 * x.expandDims(axis).print();
 * ```
 *
 * @param x The input tensor whose dimensions to be expanded.
 * @param axis The dimension index at which to insert shape of `1`. Defaults
 *     to 0 (the first dimension).
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function expandDims_<T extends Tensor>(x: Tensor|TensorLike, axis = 0): T {
  const $x = convertToTensor(x, 'x', 'expandDims', 'string_or_numeric');

  util.assert(axis <= $x.rank, () => 'Axis must be <= rank of the tensor');

  const inputs: ExpandDimsInputs = {input: $x};
  const attrs: ExpandDimsAttrs = {dim: axis};

  return ENGINE.runKernel(
      ExpandDims, inputs as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const expandDims = op({expandDims_});
